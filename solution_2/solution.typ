#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true)
#let title = [Assignment 2 -- Deconstructing the Diffusion Transformer]
#let author = [Ludwig Neste]


#show raw.where(block: true): it => {
  set text(size: 0.7em)
  set par(justify: false)
  grid(
    columns: (auto, auto),
    column-gutter: 0.3em,
    row-gutter: 0.5em,
    align: (right, left),
    ..it.lines.map(line => (
      text(fill: gray, str(line.number)),
      line,
    )).flatten()
  )
}

#let pad3(n) = {
    let s = str(n)
    if n < 10 { "00" + s } else if n < 100 { "0" + s } else { s }
}
= #title
#author
#emph[April 21, 2026]


== Task 1
_Implement the `AdaLayerNorm` class in model.py_

#figure(
```python
class AdaLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2*dim)

    def forward(self, x, cond):
        scale_shift = self.proj(cond)

        # None inserts a size-one axis, so it matches the shape of x
        scale, shift = scale_shift[:,None,:self.dim], scale_shift[:,None,self.dim:]
        
        return (1+scale)*self.norm(x) + shift
```,
caption: [The implemented `AdaLayerNorm` class following the instructions. For compactness, I removed the docstrings.]

)

=== (a) _Explain why modulating a normalized activation is strictly more expressive than adding a time embedding as in DDPM._
Effectively, only adding an embedded conditional would correspond to having only the shift term, so in that sense also 
multiplying the token by a scale term derived from the condition is doing strictly more than just adding.


=== (b) _Explain why the formulation uses (1+γ) rather than γ alone._
As per `pytorch`'s documentation, the parameters of `nn.Linear` are, by default, initialized uniformly between $plus.minus 1 \/ sqrt("cond_dim")$.
For any realistic large embedding dimension the conditioning has, that effectively means that it is initialized very close to 0. 
That would mean that at initialization the normalized token has very little effect on the layer's output, which is suboptimal for training.
Another way to see this: It resembles the same 'trick' we use when applying skip-connections, as we effectively applying the scale/shift 
to a skipped, layer-normalized token.

== Task 2

=== (a) _Intra-patch propagation_
In line `17` of @code:patch_emb, the patch gets projected into the embedding dimension.
This operation $(p_"emb" = M dot p + b)$, mixes the information within a given patch.
The second time it happens in line `17` of @code:DiTBlock, in which the token is passed 
through a small MLP, which also mixes information in the token.
Technically, in the multi-head attention step (line `15` in @code:DiTBlock), since no masking is applied,
the token also mixes information to some degree within itself#footnote([An easy way to see this is to realize that MultiheadAttention with a single token, is equivalent to a linear layer without bias.]),
although the most important function of it is obviously to mix information between tokens.

#figure(
```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(B, self.num_patches, self.patch_dim)
        x = self.proj(x)
        x = x + self.pos_embed
        return x
```,
caption: [The patch embedding class, to reference line numbers.]
)<code:patch_emb>

#figure(
```python
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x, cond))
        return x
```,
caption: [The `DiTBlock` class, to reference line numbers.]
)<code:DiTBlock>

=== (b) Inter-patch propagation
Information between tokens is moved by the multihead attention (MHA) mechanism in @code:DiTBlock (line `15`).
A $3 times 3$ convolution would need at least 9 successive layers (without padding or pooling) to move (some) information 
from the top left of the $27 times 27$ pixel MNIST-images, to the bottom right, whereas 
with MHA mechanism the top left patch is allowed to attend to the lowest right patch in the first MHA layer. 

=== (c) Advantages over the U-Net.
In a U-Net architecture, the network must be sufficiently deep to capture the full information in the image 
(as explained in (b)), the depth must be appropriate for the size of the image. 
In a `DiT`, information from across the whole image can be used from the first layer, 
so that it is reasonable to assume that the network can use global information in the image 
earlier and easier. 

The `DiT` architecture also captures both local information (intra-patch, (a)) and global information (inter-patch)
explicitly, while a U-Net design focuses more on local information in each layer.

#text([Need to shorten this by 50 words!], fill: red)
#text([Make image?], fill: red)


== Task 3

#figure(
```python
for t in tqdm(reversed(range(T)), total=T, desc="Sampling", leave=False):
    t_batch = torch.full((N,), t, device=device, dtype=torch.long)

    eps_uncond = model(x, t_batch, null_labels)
    eps_cond = model(x, t_batch, labels)

    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    sqrt_recip_alpha = sch["sqrt_recip_alphas"][t]
    beta_fac = sch["betas"][t]/sch["sqrt_one_minus_alphas_cumprod"][t]

    mean = sqrt_recip_alpha * (x - beta_fac * eps)

    if t > 0:
        z = torch.randn_like(x)
        x = mean + torch.sqrt(sch["posterior_variance"][t]) * z
    else:
        x = mean
```, 
caption: [The completed sampling loop of `sample.py`.]
)

#figure(
  image("task3_row.png", width: 50%),
  caption: [10 sampled images for each digit/class. ]
)

== Task 4
#figure(
  image("sweep.png", width: 50%),
  caption: [The guidance sweep.]
)<fig:guidance_sweep>

#figure(
  image("memorization.png", width: 20%),
  caption: [Sampled images for each digit with guidance scale $w=3.0$ (right) and the closest corresponding image in the training data (left).]
)

=== (a) How do the samples change as $w$ increases?
The unguided sample in @fig:guidance_sweep (left, $w=0$) mostly do not resemble 
any particular digit, but look like random scribbles.
Still, for $w=1.0$, some of the digits are not looking like what they are supposed to resemble (2, 6, 7, 9).
From $w=2.0$ onwards, each digit is clearly identifiable.
For some reason, which might be coincidence, for $w=4.0$ 3 digits have visible holes (2, 5, 8).
From at least $w=2.0$ onward the form of the digit stays the same, while they 
get mostly thicker and have higher contrast, with the grey-scale fraction shrinking and 
the digits mainly consisting of purely white/black pixels.
At $w=10.0$, some of the digits are also quite noisy.

=== (b) Do the nearest-neighbor pairs look like copies or plausibly novel samples?
To me, the generated digits look reasonably different from the closest nearest-neighbor image,
especially taking into account that there are only so-many ways in which one can draw a digit.
For example digit `7`, the generated image has a little single-pixel artifact, which the closest
MNIST image does not have, which is a feature some training images (here for example '2') have.
The width of the head of the `7` is also a bit wider, while it has more change in height along the head.  
If the images were merely memorized, I would not expect both of these things to happen, as it 
somehow indicates that the model has learned an abstract representation of how the digit `7` looks like.

== Task 5

#figure(
image("pixel_pr.png", width: 70%),
caption: [The precision/recall curves.],
)<fig:pr>

The recall drops sharply after $w=2$ in @fig:pr, while the precision stays reasonably high.

== Acknowledgement 
I used AI tools to enhance the language of my writing, but did not use the generated text directly -- I only considered suggestions.
I also used AI tools to answer questions about the codebase and debugging for the added code as well to help me understand
how the FID and IS are practically implemented. 
