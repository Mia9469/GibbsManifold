"""Compare the compiled local rules with a separate NumPy implementation."""
import unittest

import numpy as np

from replicate_learning import initial_model, training_bank, train


def numpy_reference(model, bank, seed, target, steps):
    rng = np.random.RandomState(seed)
    fe, ee, ei, ii, ie = [model[key][0].copy() for key in
                          ("feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie")]
    ne, ni = fe.shape[1], ei.shape[0]
    ve, vi, re, ri = np.zeros(ne), np.zeros(ni), np.zeros(ne), np.zeros(ni)
    be, bi = np.zeros(3), np.zeros(ne)
    refractory_e, refractory_i = np.full(ne, -1), np.full(ni, -1)
    ke, ki, previous_e, previous_i, template = 0, 0, 0, 0, 0
    log_gain, gain, block_spikes = 0., 1., 0
    for t in range(1, steps + 1):
        if (t - 1) % 1000 == 0:
            template = rng.randint(0, len(bank))
        signal = bank[template, :, (t - 1) % 1000]
        be = .97 * be + .0001 * signal
        bi *= .995
        bi[ke] += previous_e
        ze = rng.randn(ne, 2)
        ve = .995 * ve + .0001 * np.sum(fe * signal[:, None], axis=0) + previous_e * ee[:, ke] + previous_i * ie[:, ki] + .002 * ze[:, 0]
        score_e = ve - .5 * gain - .02 * ze[:, 1]
        ke = int(np.argmax(score_e))
        current_e = int(score_e[ke] >= 0 and refractory_e[ke] < 0)
        if current_e:
            refractory_e[ke] = 10
            fe[:, ke] += 1e-5 * (be - fe[:, ke])
            ee[:, ke] = np.maximum(ee[:, ke] - 1e-4 * (ve + .002 * re + ee[:, ke]), 0)
            ee[ke, ke] = -.02
        refractory_e -= 1
        zi = rng.randn(ni, 2)
        vi = .995 * vi + current_e * ei[:, ke] + previous_i * ii[:, ki] + .002 * zi[:, 0]
        score_i = vi - .5 * gain - .02 * zi[:, 1]
        ki = int(np.argmax(score_i))
        current_i = int(score_i[ki] >= 0 and refractory_i[ki] < 0)
        if current_i:
            refractory_i[ki] = 10
            ei[ki] = np.maximum(ei[ki] + 1e-5 * (bi - ei[ki]), 0)
            ii[:, ki] = np.minimum(ii[:, ki] - 1e-4 * (vi + .002 * ri + ii[:, ki] + .002 * (np.arange(ni) == ki)), 0)
            ie[:, ki] = np.minimum(ie[:, ki] - 1e-4 * (ve + .002 * re + ie[:, ki]), 0)
        re[ke] += current_e
        ri[ki] += current_i
        re *= .995
        ri *= .995
        refractory_i -= 1
        block_spikes += current_e + current_i
        if t % 200 == 0:
            log_gain = np.clip(log_gain + .12 * (block_spikes / (target * .02) - 1), np.log(.03), np.log(20))
            gain, block_spikes = np.exp(log_gain), 0
        previous_e, previous_i = current_e, current_i
    return fe, ee, ei, ii, ie


class TrainingTests(unittest.TestCase):
    def test_compiled_rules_match_independent_reference(self):
        model = initial_model(143, ne=8, ni=2)
        bank = training_bank(model, 143)[:2]
        keys = ("feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie")
        compiled = train(*(model[k][0].copy() for k in keys), bank, 143, 64., np.array([0,4096]))
        reference = numpy_reference(model, bank, 143, 64., 4096)
        for result, expected in zip(compiled[:5], reference):
            np.testing.assert_allclose(result[-1], expected, atol=2e-13, rtol=2e-13)


if __name__ == "__main__":
    unittest.main()
