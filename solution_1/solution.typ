#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true)
#let title = [Assignment 1 -- Denoising Diffusion Probabilistic Models (DDPM)]
#let author = [Ludwig Neste]
#let pad3(n) = {
    let s = str(n)
    if n < 10 { "00" + s } else if n < 100 { "0" + s } else { s }
}
= #title
#author
#emph[March 3, 2026]


== Task 1
_Explain why it is crucial to encode time $t$ rather than simply passing the scalar value $t$ as an extra input channel to the model?_

Using a scalar value $t$  as an input to a DDPM, or any deep learning model with some form 
of ordered input, or time steps would not adequately capture the inherent structure of ordered data.
E.g. if $t=100$ and $t=101$ look almost the same if used in the un-encoded form, while $t=0$ and $t=1$
look very different (to any linear or softmax layers). 
Even if the model is able to learn to recognize $t=0$ and $t=100$ reliably, it would need
multiple layers and it is doubtful that this would extrapolate up to very large $t$.
Choosing sinusoidal positional encoding respects this structure inherently, 
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

The time embedding is done similar to the original paper. 
The relevant code is directly in `model.py` at the top. The computation of the power of tenthousands is expressed
via exponentials.
The only difference to the embedding is, that on top of the sinusoidal embedding,
there is a simple multi-layer perceptron, that first enlarges the embedding space four-fold
and then shrinks it to the original embedding dimension again.
This added layer of learned embedding in higher dimension can help to disentangle the position information in higher
space. 
The learned embedding is then added at every residual block in the UNet architecture. 

== Task 3 
_ Plot the training and validation loss over 50 epochs. Save generated samples every 5 epochs. _

#figure(
    grid(
        columns: 4,
        gutter: 8pt,
        ..(
            (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49).map(i => [
                #align(center)[*epoch = #i*]
                #image("./generated/epoch_" + pad3(i) + ".png", width: 100%)
            ])
        ),
    ),
    caption: [Generated samples at every 5th training-epoch.],
)

#figure(
    image("./loss_plot.png", width: 100%),
    caption: [Training and validation loss over 50 epochs.],
)


== Task 4
_ How many epoch does it take for the model to start generating recognizable images?
Is there a point where the loss continues to decrease but the generated images do not improve in
quality? Discuss the implications of this observation for training diffusion models. _

Looking at the generated samples by hand, 
at epoch 20 one can start to see what the model is trying to do, but the samples 
still look very weird and unrecognizable.
At epoch 30, the samples consist of mostly recognizable digits, with approx. 10% outliers that look very weird.
Beyond 40 epochs, the samples look very good and it seems to be mostly stable.
From these observation, I would actually conclude the opposite of what the question is proposing. 
Just by looking at it, I would say the images still drastically improve after epoch 25, 
but looking at the validation loss, it does not decrease anymore.
I would say this implies the following: When training diffusion models, it is important to look 
at the generated images to decide if the model is trained enough. The loss function itself is not 
really improving anymore, which could lead to assuming that the model is done training.
This is where external evaluation metrics beside the loss function can come in useful.


== Task 5
_ Generate 5 images. Save the intermediate denoised result x̂0 (the model’s prediction of
the final image) at t = 300, 250, 200, 150, 100, 50 and 0. At what point in the timeline does the
image become ”intelligible”? Does the content (e.g., the class of the object) change in the final 200
steps, or is that period reserved for fine-tuning textures? _
I'd say at $t=150$ one can start to have a rough idea of what the image is going to look like,
while it  becomes really clear at $t=100$.

Although $t=200$ is hard to make out on it's own, knowing the class from $t=0$, 
one can see that the class is already decided at that point. At least that is 
clearly the case for the 6 and the two 8s. 
I'd concluded that the last 200 steps are thus mainly reserved for tuning textures and removing 
unneeded noise.

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

In the following table I wrote down the FID and IS for the model.
When searching of how I should implement the IS, it seemed that estimating the uncertainty 
by splitting the generated sample is standard, (I think the original paper introducing it suggested that),
so the uncertainty indicated for the IS is coming from a split of 10.
If I take the full generated set of 1000 images, I get an IS of 7.54. 

For the FID, I was not sure at which level of the classifier to use as the features. 
I implemented it both for the Layer before the last linear layer (indicated with FID in the table)
and after the last layer, before softmax should be applied (indicated with FID before softmax in the table).
The value for the second-to-last layer seems more reasonable by value, and I believe that makes sense.
The last layer before softmax has does not need to be normalized, so the FID of that layer is somewhat meaningless. 

If both the inception score and the FID are high, it implies that the model likely is inter-class collapsed;
This means that the model correctly generates the distribution of classes and the images in each class 
look plausible (or to be precise: they produce a high probability in the classifier for one class only),
but that each generated image in each class looks essentially the same.

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
    [FID before softmax (#sym.arrow.b)], [2.88], [3.17 (CIFAR-10)],
    [IS (#sym.arrow.t)], [7.14 $plus.minus$ 0.25], [9.46 (CIFAR-10)],
    table.hline(stroke: 1.3pt + black),
))



== Acknowledgement 
I used AI tools to enhance the language of my writing, but did not use the generated text directly -- I only considered the suggestions.
I also used AI tools to answer questions about the codebase and debugging for the added code as well to help me understand 
how the FID and IS are practically implemented. 