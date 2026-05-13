import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device="cuda")
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef,
                 use_uwl=False, weight_type=0,
                 actor_weight_min=0.2, actor_weight_max=1.2,
                 value_weight_min=0.2, value_weight_max=1.2,
                 uwl_temperature=1.0, uwl_eps=1e-8,
                 uncertainty_mode="single", aleatoric_coef=0.2, epistemic_coef=1.0,
                 use_varvar_epistemic=True) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.use_uwl = use_uwl
        self.weight_type = weight_type
        self.actor_weight_min = actor_weight_min
        self.actor_weight_max = actor_weight_max
        self.value_weight_min = value_weight_min
        self.value_weight_max = value_weight_max
        self.uwl_temperature = uwl_temperature
        self.uwl_eps = uwl_eps
        self.uncertainty_mode = uncertainty_mode
        self.aleatoric_coef = aleatoric_coef
        self.epistemic_coef = epistemic_coef
        self.use_varvar_epistemic = use_varvar_epistemic
        print(f'ActorCriticAgent use_uwl: {self.use_uwl}, weight_type: {self.weight_type}')
        print(f'Actor weight range: [{self.actor_weight_min}, {self.actor_weight_max}]')
        print(f'Value weight range: [{self.value_weight_min}, {self.value_weight_max}]')
        print(f'UWL temperature: {self.uwl_temperature}, eps: {self.uwl_eps}')
        print(f'UncertaintyMode: {self.uncertainty_mode}')
        print(f'AleatoricCoef: {self.aleatoric_coef}')
        print(f'EpistemicCoef: {self.epistemic_coef}')
        print(f'UseVarVarEpistemic: {self.use_varvar_epistemic}')

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, action_dim)
        )

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _normalize_confidence(self, confidence):
        temp = self.uwl_temperature
        eps = self.uwl_eps
        flat = confidence.reshape(-1, confidence.shape[-1])
        mean = torch.mean(flat, dim=0, keepdim=True)
        std = torch.std(flat, dim=0, unbiased=False, keepdim=True)
        std = torch.clamp(std, min=eps)
        normalized = (flat - mean) / std
        scaled = torch.tanh(temp * normalized)

        actor_center = (self.actor_weight_max + self.actor_weight_min) / 2.0
        actor_half_range = (self.actor_weight_max - self.actor_weight_min) / 2.0
        actor_weights = actor_center + actor_half_range * scaled

        value_center = (self.value_weight_max + self.value_weight_min) / 2.0
        value_half_range = (self.value_weight_max - self.value_weight_min) / 2.0
        value_weights = value_center + value_half_range * scaled

        return actor_weights.reshape_as(confidence), value_weights.reshape_as(confidence)

    def _compute_weights(self, confidence):
        actor_weights, value_weights = self._normalize_confidence(confidence)
        if self.weight_type == 0:
            actor_weights = None
            value_weights = None
        elif self.weight_type == 1:
            value_weights = None
        elif self.weight_type == 2:
            actor_weights = None
        elif self.weight_type == 3:
            pass
        else:
            actor_weights = None
            value_weights = None
        return actor_weights, value_weights

    def _normalize_uncertainty(self, x, eps=None):
        if eps is None:
            eps = self.uwl_eps
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(dim=0, keepdim=True)
        std = flat.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        normalized = (flat - mean) / std
        return normalized.reshape_as(x)

    def _compute_decomposed_score(self, alea_uncertainty, epi_uncertainty):
        alea_norm = self._normalize_uncertainty(alea_uncertainty)
        epi_norm = self._normalize_uncertainty(epi_uncertainty)
        score = - self.aleatoric_coef * alea_norm - self.epistemic_coef * epi_norm
        return score.detach()

    def _weighted_mean(self, tensor, weights):
        weights = weights.to(tensor.dtype)
        total_weight = torch.clamp(weights.sum(), min=1e-6)
        return (tensor * weights).sum() / total_weight

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, confidence=None,
               alea_uncertainty=None, epi_uncertainty=None, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            actor_weights = None
            value_weights = None
            decomposed_score = None
            if self.use_uwl:
                if self.uncertainty_mode == "single":
                    # Original IUPOM branch
                    if confidence is not None:
                        actor_weights, value_weights = self._compute_weights(confidence)
                elif self.uncertainty_mode == "ensemble_decomposed":
                    # Ensemble decomposed IUPOM branch
                    assert alea_uncertainty is not None, "alea_uncertainty is required in ensemble_decomposed mode"
                    assert epi_uncertainty is not None, "epi_uncertainty is required in ensemble_decomposed mode"
                    decomposed_score = self._compute_decomposed_score(alea_uncertainty, epi_uncertainty)
                    actor_weights, value_weights = self._compute_weights(decomposed_score)
                else:
                    raise ValueError(f"Unknown uncertainty mode: {self.uncertainty_mode}")
                if actor_weights is not None:
                    actor_weights = actor_weights.squeeze(-1)
                if value_weights is not None:
                    value_weights = value_weights.squeeze(-1)

            # update value function with slow critic regularization
            if value_weights is not None:
                value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach(), weights=value_weights)
                slow_value_regularization_loss = self.symlog_twohot_loss(
                    raw_value[:, :-1], slow_lambda_return.detach(), weights=value_weights)
                # print('Using value weights in value loss')
            else:
                value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
                slow_value_regularization_loss = self.symlog_twohot_loss(
                    raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            if actor_weights is not None:
                # print('Using actor weights in policy loss')
                actor_weights = actor_weights.to(log_prob.dtype)
                policy_loss = -self._weighted_mean(log_prob * norm_advantage.detach(), actor_weights)
            else:
                policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            logger.log('ActorCritic/total_loss', loss.item())
            if actor_weights is not None:
                logger.log('ActorCritic/actor_weight_mean', actor_weights.mean().item())
            if value_weights is not None:
                logger.log('ActorCritic/value_weight_mean', value_weights.mean().item())
            if alea_uncertainty is not None:
                logger.log('ActorCritic/alea_uncertainty_mean', alea_uncertainty.mean().item())
            if epi_uncertainty is not None:
                logger.log('ActorCritic/epi_uncertainty_mean', epi_uncertainty.mean().item())
            if decomposed_score is not None:
                logger.log('ActorCritic/decomposed_score_mean', decomposed_score.mean().item())
                logger.log('ActorCritic/decomposed_score_std', decomposed_score.std(unbiased=False).item())
