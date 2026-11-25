"""Supervised, NFT and population-risk policy-gradient updates."""

from copy import deepcopy
import random
import numpy as np
import torch

from glioma.eval.idea3_pareto import hypervolume


METHODS = ('dense_sft', 'moe_sft', 'dense_pg', 'moe_pg', 'dense_nft', 'moe_nft', 'moe_linear',
           'direct_pg', 'direct_sft', 'dense_epsilon')


def clipped_trajectory_objective(log_probability, old_log_probability, advantage):
    ratio = (log_probability - old_log_probability).exp()
    terms = torch.minimum(ratio * advantage[:, None], ratio.clamp(.8, 1.2) * advantage[:, None])
    # Terminal returns multiply the sum of transition scores for one trajectory.
    return terms.sum(1).mean()


def scalar_risk(risks, preference, scales, linear=False):
    weighted = (risks / scales + .05) * torch.stack([preference, 1 - preference], -1)
    return weighted.sum(-1) if linear else .05 * torch.logsumexp(weighted / .05, -1)


def population_nft_advantage(values, alpha, batch, group):
    """Shared population-risk direction, with one detached batch-wide scale.

    Alpha is estimated on the independent risk block, as for PG. A single positive
    normalization preserves the scalar reward ordering across both objectives and
    all patients; per-objective or per-patient standardization would change it.
    NFT still optimizes a clipped reconstruction surrogate, not the PG estimator.
    """
    grouped = values.reshape(batch, group, -1)
    baseline = (grouped.sum(1, keepdim=True) - grouped) / (group - 1)
    advantage = -((grouped - baseline).reshape(len(values), -1) * alpha).sum(-1).detach()
    scale = advantage.square().mean().sqrt().clamp_min(1e-4)
    return advantage / scale, scale


@torch.no_grad()
def evaluate(policy, environment, preferences, repeats, steps, noise, k, seed):
    records, points = [], []
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for weight in preferences:
            torch.manual_seed(seed)
            ids = torch.arange(len(environment.context), device=environment.context.device).repeat_interleave(repeats)
            context = environment.context[ids]
            preference = context.new_full((len(ids),), weight)
            logits, _, _, routes = policy.sample(context, preference, steps, noise)
            indices = logits.topk(k, dim=-1).indices
            losses = environment.losses(ids, indices)
            predictions = environment.reader(environment.tokens[ids], indices).cpu().tolist() if hasattr(environment, 'reader') else None
            points.append(losses.mean(0).cpu().tolist())
            for row, patient in enumerate(ids.tolist()):
                records.append({'subject_id': environment.subject_ids[patient], 'weight': float(weight),
                                'sample': row % repeats, 'anchor_loss': float(losses[row, 0]),
                                'phenotype_loss': float(losses[row, 1]), 'indices': indices[row].cpu().tolist(),
                                'routes': routes[row].mean(0).cpu().tolist()})
                if predictions is not None:
                    for field, value, target in zip(('idh', '1p19q', 'mgmt', 'phenotype'), predictions[row],
                                                    environment.targets[patient].cpu().tolist()):
                        records[-1]['prediction_' + field] = value
                        records[-1]['target_' + field] = target
    return records, points


def build_demonstrations(environment, k, proposals, local_steps):
    n, device = len(environment.context), environment.context.device
    masks = torch.randn(n, proposals, environment.items, device=device).topk(k, -1).indices
    def score(candidate):
        ids = torch.arange(n, device=device).repeat_interleave(candidate.shape[1])
        chunks = zip(ids.split(256), candidate.flatten(0, 1).split(256))
        return torch.cat([environment.losses(i, a) for i, a in chunks]).reshape(n, -1, 2)
    with torch.no_grad():
        values = score(masks)
        libraries, scores = [masks], [values]
        for weight in (0., .25, .5, .75, 1.):
            scalar = (values / environment.scales * values.new_tensor([weight, 1 - weight])).sum(-1)
            selected = masks[torch.arange(n, device=device), scalar.argmin(-1)].clone()
            best = scalar.min(-1).values
            for _ in range(local_steps):
                candidate = selected.clone()
                available = torch.ones(n, environment.items, device=device).scatter_(1, selected, 0.)
                replacement = torch.multinomial(available, 1)
                slot = torch.randint(k, (n,), device=device)
                candidate[torch.arange(n, device=device), slot] = replacement[:, 0]
                loss = environment.losses(torch.arange(n, device=device), candidate)
                scalar = (loss / environment.scales * loss.new_tensor([weight, 1 - weight])).sum(-1)
                improve = scalar < best
                selected[improve], best[improve] = candidate[improve], scalar[improve]
                libraries.append(candidate[:, None])
                scores.append(loss[:, None])
        environment.teacher_masks = torch.cat(libraries, 1)
        environment.teacher_losses = torch.cat(scores, 1)


def demonstration(environment, ids, preference, k, proposals, mode='top8'):
    with torch.no_grad():
        losses = environment.teacher_losses[ids] / environment.scales
        weights = torch.stack([preference, 1 - preference], -1)
        costs = (losses * weights[:, None]).sum(-1)
        if mode == 'best':
            eligible = costs <= costs.min(-1, keepdim=True).values + 1e-6
            chosen = torch.multinomial(eligible.float(), 1).squeeze(1)
        elif mode == 'top8':
            best = costs.topk(min(8, losses.shape[1]), largest=False).indices
            chosen = best.gather(1, torch.randint(best.shape[1], (len(ids), 1), device=ids.device)).squeeze(1)
        else:
            raise ValueError(f'Unknown demonstration mode {mode}')
        selected = environment.teacher_masks[ids, chosen]
        return torch.full((len(ids), environment.items), -1., device=ids.device).scatter_(1, selected, 1.)


def supervised_loss(policy, context, preference, target):
    if policy.direct:
        prediction = policy(torch.zeros_like(target), context, preference, target.new_zeros(len(target)))[0]
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, (target + 1) / 2)
    time = torch.rand(len(context), device=context.device)
    noise = torch.randn_like(target)
    state = noise * (1 - time[:, None]) + target * time[:, None]
    velocity = policy(state, context, preference, time)[0]
    return (velocity - (target - noise)).square().mean()


def rollout(policy, environment, preference, config):
    ids = torch.randint(len(environment.context), (config.batch,), device=environment.context.device)
    ids = ids.repeat_interleave(config.group)
    prefs = environment.context.new_full((len(ids),), preference)
    with torch.no_grad():
        logits, states, actions, _ = policy.sample(environment.context[ids], prefs, config.steps, config.noise)
        loss = environment.losses(ids, logits.topk(config.k, -1).indices)
    return ids, prefs, logits, states, actions, loss


def validation_metrics(policy, environment, config):
    weights = np.linspace(0, 1, 11) if config.fixed_weight is None else [config.fixed_weight]
    _, points = evaluate(policy, environment, weights, config.eval_repeats,
                         config.steps, config.noise, config.k, 2026)
    volume = hypervolume(points)
    score = volume if config.fixed_weight is None else -float(scalar_risk(
        environment.scales.new_tensor(points[0]), environment.scales.new_tensor(config.fixed_weight), environment.scales))
    if config.method.endswith('epsilon'):
        score = -points[0][0] - 100 * max(0., points[0][1] - config.epsilon)
    return volume, score


def optimize(policy, train, validation, config, callback=None, resume=None):
    nft_objective = getattr(config, 'nft_objective', 'legacy')
    if nft_objective not in ('legacy', 'population_stch'):
        raise ValueError('nft_objective must be legacy or population_stch')
    if nft_objective != 'legacy' and not config.method.endswith('nft'):
        raise ValueError('population_stch NFT objective requires an NFT method')
    build_demonstrations(train, config.k, config.proposals, config.local_steps)
    reference = deepcopy(policy).requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=1e-4)
    best, best_state, history = -float('inf'), None, []
    dual = 0.
    start = 0
    if config.warmup == 0 and resume is None:
        volume, best = validation_metrics(policy, validation, config)
        best_state = deepcopy(policy.state_dict())
        history.append({'update': 0, 'loss': None, 'validation_hv': volume,
                        'selection_score': best, 'diagnostics': {'phase': 'initial_checkpoint'}})
    if resume is not None:
        policy.load_state_dict(resume['model'])
        reference.load_state_dict(resume['reference'])
        optimizer.load_state_dict(resume['optimizer'])
        start, best, best_state, history = resume['update'], resume['best_score'], resume['best_state'], resume['history']
        torch.set_rng_state(resume['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in resume['cuda_rng']])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        dual = resume.get('dual', 0.)
    for update in range(start, config.warmup + config.updates):
        diagnostics = {}
        preference = float(np.random.uniform())
        if np.random.uniform() < .1:
            preference = float(np.random.randint(2))
        if config.fixed_weight is not None and update >= config.warmup:
            preference = config.fixed_weight
        if update < config.warmup or config.method.endswith('sft'):
            ids = torch.randint(len(train.context), (config.batch,), device=train.context.device)
            prefs = train.context.new_full((len(ids),), preference)
            target = demonstration(train, ids, prefs, config.k, config.proposals, config.teacher_mode)
            loss = supervised_loss(policy, train.context[ids], prefs, target)
        else:
            # Independent block estimates the population objective before scalarization.
            _, _, _, _, _, estimate = rollout(policy, train, preference, config)
            risks = estimate.mean(0).detach().requires_grad_()
            alpha = torch.autograd.grad(scalar_risk(risks, risks.new_tensor(preference), train.scales,
                                                   config.method.endswith('linear')), risks)[0].detach()
            if config.method.endswith('epsilon'):
                if config.epsilon is None:
                    raise ValueError('Epsilon-constraint training requires --epsilon')
                dual = max(0., dual + .1 * (float(risks[1].detach()) - config.epsilon))
                alpha = risks.new_tensor([1 / float(train.scales[0]), dual])
            ids, prefs, clean, states, actions, values = rollout(policy, train, preference, config)
            diagnostics['population_risk'] = estimate.mean(0).cpu().tolist()
            diagnostics['risk_coefficients'] = alpha.cpu().tolist()
            diagnostics['within_group_reward_std'] = values.reshape(config.batch, config.group, 2).std(
                1, unbiased=False).mean(0).cpu().tolist()
            if config.method.endswith('nft'):
                if nft_objective == 'population_stch':
                    advantages, advantage_scale = population_nft_advantage(
                        values, alpha, config.batch, config.group)
                    diagnostics['nft_objective'] = nft_objective
                    diagnostics['scalar_advantage_scale'] = float(advantage_scale)
                    diagnostics['scalar_advantage_std'] = float(advantages.std(unbiased=False))
                    diagnostics['advantage_clip_fraction'] = float((advantages.abs() > 5).float().mean())
                else:
                    rewards = (-values / train.scales).reshape(config.batch, config.group, 2)
                    advantages = ((rewards - rewards.mean(1, keepdim=True)) /
                                  (rewards.std(1, keepdim=True, unbiased=False) + 1e-4)).reshape(-1, 2)
                time = torch.rand(len(ids), device=ids.device)
                noise = torch.randn_like(clean)
                state = noise * (1 - time[:, None]) + clean * time[:, None]
                prediction = policy(state, train.context[ids], prefs, time)[0]
                with torch.no_grad():
                    # One online update per rollout, so the behavior snapshot is current theta.
                    old = prediction.detach()
                    reference_prediction = reference(state, train.context[ids], prefs, time)[0]
                positive = state + (1 - time[:, None]) * (.1 * prediction + .9 * old)
                negative = state + (1 - time[:, None]) * (1.1 * old - .1 * prediction)
                positive_loss = ((positive - clean).square() /
                                 (positive - clean).abs().mean(-1, keepdim=True).detach().clamp_min(1e-5)).mean(-1)
                negative_loss = ((negative - clean).square() /
                                 (negative - clean).abs().mean(-1, keepdim=True).detach().clamp_min(1e-5)).mean(-1)
                reward_weight = .5 + .5 * (advantages / 5).clamp(-1, 1)
                if nft_objective == 'population_stch':
                    loss = (reward_weight * positive_loss + (1 - reward_weight) * negative_loss).mean() / .1 * 5
                else:
                    per_objective = (reward_weight * positive_loss[:, None] +
                                     (1 - reward_weight) * negative_loss[:, None]) / .1
                    loss = (per_objective * torch.stack([prefs, 1 - prefs], -1)).sum(-1).mean() * 5
                loss = loss + 1e-4 * (prediction - reference_prediction).square().mean()
            else:
                grouped = values.reshape(config.batch, config.group, 2)
                baseline = (grouped.sum(1, keepdim=True) - grouped) / (config.group - 1)
                advantage = -((grouped - baseline).reshape(-1, 2) * alpha).sum(-1).detach()
                context = train.context[ids]
                old_log = policy.log_probability(states, actions, context, prefs, config.noise).detach()
                log_probability = policy.log_probability(states, actions, context, prefs, config.noise)
                objective = clipped_trajectory_objective(log_probability, old_log, advantage)
                trajectory_steps = states.shape[1]
                times = torch.arange(trajectory_steps, device=ids.device).expand(len(ids), -1) / trajectory_steps
                contexts = context[:, None].expand(-1, trajectory_steps, -1).flatten(0, 1)
                preferences = prefs[:, None].expand(-1, trajectory_steps).flatten()
                velocity = policy(states.flatten(0, 1), contexts, preferences, times.flatten())[0]
                with torch.no_grad():
                    reference_velocity = reference(states.flatten(0, 1), contexts, preferences, times.flatten())[0]
                if policy.direct:
                    available = torch.ones_like(velocity, dtype=torch.bool)
                    kl_terms = []
                    for choice in actions.unbind(1):
                        current_distribution = torch.distributions.Categorical(logits=velocity.masked_fill(~available, -1e9))
                        reference_distribution = torch.distributions.Categorical(logits=reference_velocity.masked_fill(~available, -1e9))
                        kl_terms.append(torch.distributions.kl_divergence(current_distribution, reference_distribution))
                        available = available.scatter(1, choice[:, None], False)
                    kl = torch.stack(kl_terms).sum(0).mean()
                else:
                    kl = (velocity - reference_velocity).square().sum(-1).mean() / (2 * config.steps * config.noise ** 2)
                loss = -objective + 1e-4 * kl
                diagnostics['advantage_std'] = float(advantage.std(unbiased=False))
                diagnostics['reference_kl'] = float(kl.detach())
                diagnostics['trajectory_steps'] = trajectory_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at update {update}')
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        diagnostics['gradient_norm_before_clip'] = float(gradient_norm)
        optimizer.step()
        if update + 1 == config.warmup:
            reference.load_state_dict(policy.state_dict())
        if (update + 1) % config.validate_every == 0 or update + 1 == config.warmup + config.updates:
            volume, score = validation_metrics(policy, validation, config)
            history.append({'update': update + 1, 'loss': float(loss.detach()), 'validation_hv': volume,
                            'selection_score': score, 'diagnostics': diagnostics})
            if score > best:
                best, best_state = score, deepcopy(policy.state_dict())
            if callback:
                callback(update + 1, policy, optimizer, reference, history, best, best_state, dual)
            print(f'{config.method} update={update+1} loss={float(loss.detach()):.5f} val_hv={volume:.6f}', flush=True)
    policy.load_state_dict(best_state)
    return history
