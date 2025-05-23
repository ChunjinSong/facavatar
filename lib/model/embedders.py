import torch

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        min_freq = self.kwargs['min_freq']
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(min_freq, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**min_freq, 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3, mode='fourier', min_freq=0., include_input=True):
    embed_kwargs = {
        'include_input': include_input,
        'input_dims': input_dims,
        'min_freq': min_freq,
        'max_freq_log2': min_freq + multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    if mode == 'fourier':
        embedder_obj = Embedder(**embed_kwargs)


    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim