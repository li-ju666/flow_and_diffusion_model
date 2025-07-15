import torch


class Simulator:
    def __init__(self, prob_path):
        self.prob_path = prob_path

    def simulate(self, y, num_steps=20):
        eps = 1e-7
        ts = torch.linspace(eps, 1, num_steps).view(-1, 1, 1) # batch size x 1 x 1
        dt = ts[1] - ts[0] # assuming uniform time steps
        dt = dt.repeat(y.shape[0], 1) # batch size x 1 x 1

        x = self.prob_path.sample_x0(y.shape[0]) # batch size x dim
        for t in ts:
            t = t.repeat(x.shape[0], 1)
            x = self.step(x, y, t, dt) # batch size x dim
        return x

    def step(self, x, y, t, dt):
        pass


class ODESimulator(Simulator):
    @torch.no_grad()
    def step(self, x, y, t, dt):
        # print(x.shape, y.shape, t.shape, dt.shape)
        u = self.prob_path.model(x, y, t)
        return x + u * dt


class CFODESimulator(Simulator):
    @torch.no_grad()
    def step(self, x, y, t, dt):
        m = 5
        # compute the guided vector field
        guided_u = self.prob_path.model(x, y, t)

        # compute the unguided vector field
        y = torch.ones_like(y) * 10
        unguided_u = self.prob_path.model(x, y, t)

        return x + ((1-m)* unguided_u + m * guided_u) * dt