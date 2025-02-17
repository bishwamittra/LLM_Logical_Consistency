# dataset="NELL"
dataset="FB15k"
python prepare_intersection_union_FB15k.py test 1u2i --dataset ${dataset}
python prepare_intersection_union_FB15k.py test 1i2u --dataset ${dataset}
python prepare_intersection_union_FB15k.py valid 1u2i --dataset ${dataset}
python prepare_intersection_union_FB15k.py valid 1i2u --dataset ${dataset}
