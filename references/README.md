The example ckpt in this folder is the sqlite portion of the Draw Things model
formatit's for flux_2_klein_4b_q6p

These tiny ckpt's are formed naturally by drawthings when you"enable faster
loading" by clicking on the 3 dots in the model list that pops up when you
select "Manage" in the model picker dropdown. The tensor data goes to file named
the same but with `-tensordata` appended

The custom_lora.json and custom.json are just for messing around testing it.

- safe-swap.py is the file i used to do the one LoRa fix initially, it swaps the
  incorrect key names for the right ones

- fix-safetensors-headers.py is what i was using to fix metadata issues inside
  the file.I just stripped it all, seems fine, DT doesnt preserve it anyways

Hence the reason for the entire thing lol
