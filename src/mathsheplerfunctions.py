# =====================================================================
# 2. MATHEMATICAL HELPER FUNCTIONS
# =====================================================================
def get_vector_length(vector_dict):
    sum_of_squares = sum(value ** 2 for value in vector_dict.values())
    return math.sqrt(sum_of_squares)

def calculate_cosine_similarity(user_vector, target_vector):
    dot_product = sum(user_vector.get(topic, 0.0) * target_vector.get(topic, 0.0) 
                      for topic in target_vector)
            
    user_length = get_vector_length(user_vector)
    target_length = get_vector_length(target_vector)
    
    if user_length == 0 or target_length == 0:
        return 0.0
    return dot_product / (user_length * target_length)
