import torch


class Simulator:
    def __init__(self, prob_path):
        self.prob_path = prob_path

    def simulate(self, y, num_steps=20):
        ts = torch.linspace(0, 1, num_steps).view(-1, 1, 1) # batch size x 1 x 1
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
        u = self.prob_path.vector_field_guided_y(x, y, t)
        return x + u * dt


class SDESimulator(Simulator):
    def __init__(self, prob_path, sigma=1.0):
        super().__init__(prob_path)
        self.sigma = sigma

    @torch.no_grad()
    def step(self, x, y, t, dt):
        u = self.prob_path.vector_field_guided_y(x, y, t)
        s = self.prob_path.score_guided_y(x, y, t, u)
        # check if nan exist in s
        if torch.isnan(s).any():
            raise ValueError("NaN detected in score")
        if torch.isnan(u).any():
            raise ValueError("NaN detected in vector field")
        t1 = u + self.sigma**2/2 * s
        if torch.isnan(t1).any():
            raise ValueError("NaN detected in t1")
        t2 = self.sigma * torch.sqrt(dt) * torch.randn_like(x)
        if torch.isnan(t2).any():
            raise ValueError("NaN detected in t2")
        x = x + t1 * dt + t2
        return x
