#set page(paper: "a4", margin: 2.5cm)
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true)
#let title = [Assignment 2 -- Deconstructing the Diffusion Transformer]
#let author = [Ludwig Neste]
#let pad3(n) = {
    let s = str(n)
    if n < 10 { "00" + s } else if n < 100 { "0" + s } else { s }
}
= #title
#author
#emph[April 21, 2026]


== Task 1
_Task description_

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


== Acknowledgement 
I used AI tools to enhance the language of my writing, but did not use the generated text directly -- I only considered suggestions.
I also used AI tools to answer questions about the codebase and debugging for the added code as well to help me understand
how the FID and IS are practically implemented. 
