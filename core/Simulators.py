import torch
import torch.nn.functional as F


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
    def step(self, x, y, t, dt):
        m = 5
        # compute the guided vector field
        with torch.no_grad():
            u, _ = self.prob_path.model(x, t)

        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                film = self.prob_path.model.time_embed(t)
                logits = self.prob_path.model.classifier(x_in, film)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in)[0]

        guidance = cond_fn(x, t, y)
        # compute the vector field
        vector_field = u + guidance * 4

        # vector_field = u
        # update the state
        return x + vector_field * dt
