class Potential:
    def __init__(self, potential, cutoff, effective_cutoff=None):
        self.potential = potential
        self.cutoff = cutoff

        if effective_cutoff is None:
            self.effective_cutoff = cutoff
        else:
            self.effective_cutoff = effective_cutoff

    def __call__(self, graph):
        # signature: Graph -> ndarray (energies per atom)
        return self.potential(graph)
