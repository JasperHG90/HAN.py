# HAN.py

This python module contains an implementation of a Hierarchical Attention Network (HAN). You can find details on the model in the following paper:

>Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical attention >networks for document classification. In Proceedings of the 2016 conference of the North American >chapter of the association for computational linguistics: human language technologies (pp. >1480-1489).

The paper can be found [here](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf). 
 
Through its use of the Attention mechanism, the HAN can  retain  some  of  the  hierarchical structure inherent to textual data. The attention mechanism allows us to assign weights to each word based on their importance. Hence, we can pick out the most ’informative’ words of a sentence, as  well  as  the  most  informative  sentences  in  a  document. Therefore,  we  expect  the  model  to  be  somewhat  ’context-aware’. 

The  HAN  consists  of  five  separate  modules (see image below. Image is taken from the paper by Yang et al.). 

<img src="img/attn.png" width=250></img>

First, we feed the input sequences to a word encoder, which is a bidirectional Gated Recurrent Unit (GRU). Like the LSTM, the GRU is a recurrent neural network  that  allows  us  to  carry  information  across  long sequences  of  input  data.  However,  the  architecture  of  the GRU is simpler than the LSTM and as such is considerably faster. By using a bidirectional GRU, we can use informationby scanning the sequence from left to right and vice versa.

We  apply  attention  to  each  of  the  intermittent  hidden states  to  obtain  a  sentence  vector  for  each  sentence.  The sentence vectors are then concatenated together. This serves as  the  input  to  the  sentence  encoder  (also  a  bidirectional GRU). We again apply the attention mechanism. The output of this process is fed to a softmax classifier that predicts the topic of the document. See image below for a schematic overview of the implementation in this module:

<img src="img/implementation.png"></img>


