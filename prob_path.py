class ProbabilityPath:
    def __init__(self, p_init, alpha, beta):
        self.p_init = p_init
        self.alpha = alpha
        self.beta = beta

    def sample_x(self, z, t):
        # Sample x given z and t
        pass

    def reference_vector_field(self, x, z, t):
        # Compute the reference vector field at time t
        pass


class GaussianPath(ProbabilityPath):
    def __init__(self, alpha, beta):
        p_init = None
        super().__init__(p_init, alpha, beta)

    def sample_x(self, z, t):
        # Sample x from a Gaussian distribution
        epsilon = torch.randn_like(z)
        return z * self.alpha(t) + self.beta(t) * epsilon

    def reference_vector_field(self, x, z, t):
        t1 = self.alpha.dt(t) - self.alpha(t) * self.beta.dt(t) / self.beta(t)
        t2 = self.beta.dt(t) / self.beta(t)
        return t1 * x + t2 * z
