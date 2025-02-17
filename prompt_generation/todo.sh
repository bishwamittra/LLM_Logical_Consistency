dataset="FB15k"
# dataset="NELL"
# dataset="wiki"
nrows=10000

# 1c p, \neg p
python prompt_generation_negation_query.py ../data_optimized/1c_data_final_${dataset}_train_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_negation_query.py ../data_optimized/1c_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_negation_query.py ../data_optimized/1c_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows


# 2i p \wedge q
python prompt_generation_logic_queries.py ../data_optimized/2i_data_final_${dataset}_train_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_logic_queries.py ../data_optimized/2i_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_logic_queries.py ../data_optimized/2i_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows

# 2u p \vee q
python prompt_generation_logic_queries.py ../data_optimized/2u_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_logic_queries.py ../data_optimized/2u_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows

# # 1i2u p \wedge (q \vee r)
python prompt_generation_logic_queries.py ../data_optimized/1i2u_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_logic_queries.py ../data_optimized/1i2u_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows


# # 1u2i p \vee (q \wedge r)
python prompt_generation_logic_queries.py ../data_optimized/1u2i_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_logic_queries.py ../data_optimized/1u2i_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows


# 2i commutative: p \wedge q = q \wedge p
python prompt_generation_commutative_property.py ../data_optimized/2i_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_commutative_property.py ../data_optimized/2i_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows


# 2u commutative: p \vee q = q \vee p
python prompt_generation_commutative_property.py ../data_optimized/2u_data_final_${dataset}_test_2.csv --context_length 1000 --nrows $nrows
python prompt_generation_commutative_property.py ../data_optimized/2u_data_final_${dataset}_valid_2.csv --context_length 1000 --nrows $nrows