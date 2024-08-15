"""
@inproceedings{Otmazgin2022FcorefFA,
  title={F-coref: Fast, Accurate and Easy to Use Coreference Resolution},
  author={Shon Otmazgin and Arie Cattan and Yoav Goldberg},
  booktitle={AACL},
  year={2022}
}
# https://github.com/shon-otmazgin/fastcoref
"""



from fastcoref import LingMessCoref

model = LingMessCoref(device='cuda:0')

preds = model.predict(
   texts=['We are so happy to see you using our coref package. This package is very fast!']
)

spans = preds[0].get_clusters(as_strings=False)

chains = preds[0].get_clusters()

logit = preds[0].get_logit(
   span_i=(33, 50), span_j=(52, 64)
)

print(spans)
print(chains)
print(logit)