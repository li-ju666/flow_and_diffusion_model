import torch


class Simulator:
    def __init__(self, model, prob_path):
        self.model = model # neural network model
        self.model.eval()
        self.prob_path = prob_path

    def simulate(self, y, num_steps=20):
        ts = torch.linspace(0, 1, num_steps).view(-1, 1, 1) # batch size x 1 x 1
        dt = ts[1] - ts[0] # assuming uniform time steps
        dt = dt.repeat(y.shape[0], 1) # batch size x 1 x 1
        x = self.prob_path.sample_p_init(y.shape[0]) # batch size x dim
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
        u = self.model(x, y, t)
        return x + u * dt
