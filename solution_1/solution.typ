#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true)
#let title = [Assignment 1 -- Denoising Diffusion Probabilistic Models (DDPM)]
#let author = [Ludwig Neste]
= #title
#author
#emph[March 3, 2026]


== Task 1
_Explain why it is crucial to encode time $t$ rather than simply passing the scalar value $t$ as an extra input channel to the model?_

Using a scalar value $t$  as an input to a DPPM, or any deep learning model with some form 
of ordered input, or time steps would not adequatly capture the inherit structure of ordered data.
E.g. if $t=100$ and $t=101$ look almost the same if used in the un-encoded form, while $t=0$ and $t=1$
look very different (to any linear or softmax layers). 
Even if the model is able to learn to recognize $t=0$ and $t=100$ reliably, it would need
multiple layers and it is doutable that this would extrapolate up to very large $t$.
Choosing sinusoidal positional encoding respects this structure inherintly, 
since it produces output with a maximum normalization of $1$, regardless of $t$ and
also makes it very easy to learn the relative position, since the give formulas obey 

$P E(t+k, 2 i) 
// = sin((t+k) / (10 thin 000^(2 i slash d_("model" ) )))
// = sin(t / (10 thin 000^(2 i slash d_("model" ) ))) 
//   cos(k / (10 thin 000^(2 i slash d_("model" ) ))) 
//   + cos(t / (10 thin 000^(2 i slash d_("model" ) ))) 
//   sin(k / (10 thin 000^(2 i slash d_("model" ) ))) 
= P E(t, 2 i) P E(k, 2i+1)+P E(t, 2 i+1) P E(k, 2i)
$

and 

$P E(t+k, 2 i+1) 
// = sin((t+k) / (10 thin 000^(2 i slash d_("model" ) )))
// = sin(t / (10 thin 000^(2 i slash d_("model" ) ))) 
//   cos(k / (10 thin 000^(2 i slash d_("model" ) ))) 
//   + cos(t / (10 thin 000^(2 i slash d_("model" ) ))) 
//   sin(k / (10 thin 000^(2 i slash d_("model" ) ))) 
= P E(t, 2 i+1) P E(k, 2i+1)-P E(t, 2 i) P E(k, 2i).
$

This means to learn the relative distance $k$ of $t$ and $t+k$, 
the model only needs to learn the right linear combination of the two encodings. 

== Task 2
_ Explain how the time t is embedded in to the UNet model in the provided code. That is,
go through the code and identify where and how the positional encoding is computed and injected
into the model. _

The time embeding is done similar to the original paper. 
The relevant code is directly in `model.py` at the top. The computation of the power of tenthousands is expressed
via exponentials.
The only difference to the embeding is, that on top of the sinusoidal embedding,
there is a simple multi-layer perceptron, that first enlargens the embedding space four-fold
and then shrinks it to the original embeding dimension again.
This added layer of learned embeding in higher dimension can help to disentangle the position information in higher
space. 
The learned embedinng is then added at every risidual block in the UNet architecture. 

== Task 3 
_ Plot the training and validation loss over 50 epochs. Save generated samples every 5 _
#figure(
    image("./loss_plot.png", width: 100%),
    caption: [Training and validation loss over 50 epochs.],
)

== Task 4
_ How many epoch does it take for the model to start generating recognizable images?
Is there a point where the loss continues to decrease but the generated images do not improve in
quality? Discuss the implications of this observation for training diffusion models.
_

== Task 5

#let pad3(n) = {
    let s = str(n)
    if n < 10 { "00" + s } else if n < 100 { "0" + s } else { s }
}

#figure(
    grid(
        columns: 3,
        gutter: 8pt,
        ..(
            (0, 50, 100, 150, 200, 250, 300).rev().map(i => [
                #align(center)[*t = #i*]
                #image("./generated/" + pad3(i) + ".png", width: 100%)
            ])
        ),
    ),
    caption: [Generated samples at steps $t=$ 0, 50, 100, 150, 200, 250 and 300.],
)

== Task 6
_ Compute the FID and IS for 1,000 generated samples. If your model achieves a high
Inception Score but also a high FID, what does this suggest about the diversity of your generated
samples relative to the true dataset? _ 

// center this
#align(center,table(
    columns: 3,
    // width: 100%,
    align: (left, center, center),
    stroke: none,
    table.hline(stroke: 1.3pt + black),
    table.header([Metric], [Model], [Baseline (Ho et al.)]),
    table.hline(stroke: 0.8pt + black),
    [FID (#sym.arrow.b)], [5.8], [3.17 (CIFAR-10)],
    [FID (#sym.arrow.b) (BEFORE SOFTMAX)], [2.88], [3.17 (CIFAR-10)],
    [IS (#sym.arrow.t)], [7.14 $plus.minus$ 0.25], [9.46 (CIFAR-10)],
    table.hline(stroke: 1.3pt + black),
))



== Acknowledgement 
I used AI tools to enhance the language of my writing, but did not use the generated text directly -- I only considered the suggestions.
I also used AI tools to answer questions about the codebase and debugging for the added code.