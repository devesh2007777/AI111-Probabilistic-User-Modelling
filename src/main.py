import math
import time
import json
import google.generativeai as genai

# =====================================================================
# 1. SETUP LLM API (The NLP Brain)
# =====================================================================
API_KEY = "ENTER_API_KEY_HERE" 
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

def parse_with_real_ai(user_sentence, valid_categories):
    prompt = f"""
    The user said: "{user_sentence}"
    
    You are a classification engine mapping text to a STRICT predefined vector space. 
    You MUST ONLY use categories from this exact list:
    {valid_categories}
    
    Analyze the user's intent:
    - If they LIKE or WANT something, give it a POSITIVE weight (0.1 to 1.0).
    - If they DISLIKE or NEGATE something, give it a NEGATIVE weight (-0.1 to -1.0).
    
    Return ONLY a valid JSON dictionary with the top 2 to 4 impacted categories and their float weights. 
    DO NOT output any markdown, explanations, or text outside the JSON object.
    """
    try:
        response = llm_model.generate_content(prompt)
        clean_text = response.text.strip().strip('json').strip('').strip()
        impact_weights = json.loads(clean_text)
        strict_weights = {k: v for k, v in impact_weights.items() if k in valid_categories}
        return strict_weights
    except Exception as e:
        print(f"\n[System Notice: The NLP Engine couldn't parse that sentence cleanly.]")
        return {}

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

# =====================================================================
# 3. CORE AI ENGINE: PROBABILISTIC USER MODEL
# =====================================================================
class ContextAwareUserModel:
    def _init_(self, user_id, initial_long_term_stats):
        self.user_id = user_id
        self.long_term = initial_long_term_stats.copy()  
        self.short_term = initial_long_term_stats.copy() 
        
        # --- TUNED HYPERPARAMETERS ---
        self.alpha_short = 0.85   # Fast Weather: Highly reactive to new input
        self.alpha_long = 0.15    # Fast Climate: Shifts into personalities quickly
        self.forget_threshold = 0.05 
        
        self.weight_short = 0.70  
        self.weight_long = 0.30   
        
        self.character_archetypes = {
            "The Hardcore Gamer": {"Gaming_FPS": 0.9, "Gaming_RPG": 0.9, "Beverages_EnergyDrink": 0.8, "FastFood_Pizza": 0.7},
            "The Gym Rat": {"Fitness_Weights": 0.9, "Fitness_Cardio": 0.8, "Wellness_Vitamins": 0.9, "FastFood_Chicken": 0.7},
            "The Zen Master": {"Wellness_Meditation": 0.9, "Fitness_Yoga": 0.9, "CoffeeShop_Tea": 0.8, "Books_SelfHelp": 0.7},
            "The Tech Bro": {"Tech_Laptop": 0.9, "Tech_Smartwatch": 0.8, "Education_Coding": 0.9, "CoffeeShop_Espresso": 0.8},
            "The Cinephile": {"Cinema_Drama": 0.9, "Cinema_Action": 0.8, "Streaming_HBO": 0.9, "Books_Biography": 0.6},
            "The Party Animal": {"Music_EDM": 0.9, "Beverages_Beer": 0.9, "FastFood_Burger": 0.8, "Travel_City": 0.7},
            "The Outdoorsman": {"Travel_Mountain": 0.9, "Travel_Camping": 0.9, "Hobbies_Photography": 0.7, "Cars_Truck": 0.8},
            "The Homebody": {"Streaming_Netflix": 0.9, "Home_Furniture": 0.8, "Pets_Cat": 0.9, "Hobbies_Cooking": 0.7},
            "The Hustler": {"CoffeeShop_ColdBrew": 0.9, "Education_Math": 0.8, "Tech_Smartphone": 0.9, "Books_SelfHelp": 0.8},
            "The Foodie": {"FineDining_Steak": 0.9, "FineDining_Sushi": 0.9, "FineDining_Wine": 0.8, "Travel_City": 0.7},
            "The Athlete": {"Sports_Soccer": 0.9, "Sports_Basketball": 0.9, "Fitness_Cardio": 0.8, "Beverages_Water": 0.9},
            "The Fashionista": {"Fashion_Streetwear": 0.9, "Fashion_Formal": 0.8, "Home_Decor": 0.7, "Travel_City": 0.8},
            "The Mechanic": {"Cars_SportsCar": 0.9, "Cars_SUV": 0.8, "Hobbies_DIY": 0.9, "Tech_Smartphone": 0.6},
            "The Academic": {"Education_Science": 0.9, "Education_History": 0.9, "Books_Biography": 0.8, "CoffeeShop_Tea": 0.8},
            "The Musician": {"Music_Rock": 0.9, "Music_Classical": 0.8, "Tech_Headphones": 0.9, "Hobbies_DIY": 0.6},
            "The Animal Lover": {"Pets_Dog": 0.9, "Pets_Cat": 0.9, "Travel_Mountain": 0.7, "Hobbies_Photography": 0.7},
            "The Backpacker": {"Travel_Beach": 0.9, "Travel_Cruise": 0.8, "Fashion_Activewear": 0.8, "Beverages_Beer": 0.7},
            "The Creative": {"Hobbies_Painting": 0.9, "Hobbies_Photography": 0.9, "Music_Pop": 0.7, "Home_Decor": 0.8},
            "The Data Nerd": {"Education_Coding": 0.9, "Gaming_Strategy": 0.9, "Tech_Laptop": 0.8, "Books_SciFi": 0.8},
            "The Health Nut": {"Wellness_Skincare": 0.9, "Wellness_Sleep": 0.9, "Beverages_Water": 0.9, "Fitness_Yoga": 0.8}
        }

    def register_action(self, impact_weights_dict):
        topics_to_forget = []
        
        for topic in list(self.short_term.keys()):
            self.short_term[topic] = (1 - self.alpha_short) * self.short_term[topic]
            if self.short_term[topic] < self.forget_threshold:
                topics_to_forget.append(topic)
                
        for topic in topics_to_forget:
            del self.short_term[topic]
            
        for topic, semantic_weight in impact_weights_dict.items():
            if topic not in self.short_term:
                self.short_term[topic] = 0.0
                
            self.short_term[topic] += (self.alpha_short * semantic_weight)
            self.short_term[topic] = max(0.0, min(self.short_term[topic], 1.0))

        for topic in self.short_term.keys():
            if topic not in self.long_term:
                self.long_term[topic] = 0.0 
            self.long_term[topic] = ((1 - self.alpha_long) * self.long_term[topic]) + \
                                    (self.alpha_long * self.short_term[topic])

    def print_state(self):
        print(f"\n{'='*55}")
        print(f" USER PROFILE: {self.user_id}")
        print(f"{'='*55}")
        
        print("\n[Short-Term Weather] (Top 5 active):")
        for topic, prob in sorted(self.short_term.items(), key=lambda item: item[1], reverse=True)[:5]:
             if prob > 0.01:
                 print(f"  -> {topic.ljust(25)} : {prob:.3f}")
             
        print("\n[Long-Term Climate] (Top 5 core):")
        for topic, prob in sorted(self.long_term.items(), key=lambda item: item[1], reverse=True)[:5]:
             if prob > 0.01:
                 print(f"  -> {topic.ljust(25)} : {prob:.3f}")
        print(f"{'-'*55}")

    def predict_character(self):
        best_match = "Blank Slate (Needs Data)"
        # Tuned to strictly > 0.0 so a completely blank vector returns "Blank Slate"
        highest_score = 0.0  
        
        for character_name, vector in self.character_archetypes.items():
            score = calculate_cosine_similarity(self.long_term, vector)
            if score > highest_score:
                highest_score = score
                best_match = character_name
                
        print(f"\n🔮 PSYCHOLOGICAL PROFILE: [{best_match}]")
        if highest_score > 0.0:
            print(f"   (Confidence Match: {highest_score * 100:.1f}%)")
        print(f"{'-'*55}")

    def serve_best_ad(self):
        ad_inventory = {
            "Monday Morning Survival Espresso": {"vector": {"CoffeeShop_Espresso": 0.9, "Beverages_EnergyDrink": 0.4}, "payout": 4.50},
            "Cozy Rainy-Day Vanilla Latte": {"vector": {"CoffeeShop_Latte": 0.9, "Wellness_Sleep": 0.3}, "payout": 5.50},
            "High-Focus Cold Brew Subscription": {"vector": {"CoffeeShop_ColdBrew": 0.9, "Education_Coding": 0.6}, "payout": 15.00},
            "Sweet Treat Pastry Escape": {"vector": {"CoffeeShop_Pastry": 0.9, "Wellness_Meditation": 0.2}, "payout": 3.00},
            "Soothing Herbal Tea Blend": {"vector": {"CoffeeShop_Tea": 0.9, "Wellness_Sleep": 0.8}, "payout": 8.00},
            "Late-Night Depression Burger": {"vector": {"FastFood_Burger": 0.9, "Streaming_Netflix": 0.7}, "payout": 12.00},
            "Party-Time Pizza Feast": {"vector": {"FastFood_Pizza": 0.9, "Gaming_FPS": 0.6}, "payout": 25.00},
            "Stress-Eating Loaded Fries": {"vector": {"FastFood_Fries": 0.9, "Cinema_Action": 0.5}, "payout": 6.00},
            "Late-Night Spicy Tacos": {"vector": {"FastFood_Tacos": 0.9, "Beverages_Beer": 0.6}, "payout": 14.00},
            "Comfort Fried Chicken Bucket": {"vector": {"FastFood_Chicken": 0.9, "Streaming_Hulu": 0.6}, "payout": 20.00},
            "Romantic Anniversary Steak Dinner": {"vector": {"FineDining_Steak": 0.9, "FineDining_Wine": 0.8}, "payout": 150.00},
            "Celebratory Sushi Platter": {"vector": {"FineDining_Sushi": 0.9, "Travel_City": 0.5}, "payout": 80.00},
            "Relaxing Vintage Wine Delivery": {"vector": {"FineDining_Wine": 0.9, "Home_Decor": 0.4}, "payout": 45.00},
            "Comforting Authentic Pasta": {"vector": {"FineDining_Pasta": 0.9, "Books_Romance": 0.4}, "payout": 35.00},
            "Luxurious Seafood Escape": {"vector": {"FineDining_Seafood": 0.9, "Travel_Beach": 0.6}, "payout": 120.00},
            "Hydration Recovery Water": {"vector": {"Beverages_Water": 0.9, "Fitness_Cardio": 0.8}, "payout": 2.00},
            "Sugar-Rush Nostalgia Soda": {"vector": {"Beverages_Soda": 0.9, "Gaming_RPG": 0.5}, "payout": 3.00},
            "Morning Bright Detox Juice": {"vector": {"Beverages_Juice": 0.9, "Wellness_Skincare": 0.7}, "payout": 9.00},
            "All-Nighter Monster Energy": {"vector": {"Beverages_EnergyDrink": 0.9, "Education_Coding": 0.8}, "payout": 4.00},
            "Friday Night Party Beer Case": {"vector": {"Beverages_Beer": 0.9, "Sports_Soccer": 0.7}, "payout": 22.00},
            "Anxiety-Relief Guided Meditation App": {"vector": {"Wellness_Meditation": 0.9, "Books_SelfHelp": 0.6}, "payout": 12.00},
            "Self-Care Skincare Routine Box": {"vector": {"Wellness_Skincare": 0.9, "Fashion_Casual": 0.5}, "payout": 45.00},
            "Deep Tissue Stress Massage": {"vector": {"Wellness_Massage": 0.9, "Travel_Beach": 0.4}, "payout": 80.00},
            "Immunity Boost Vitamin Pack": {"vector": {"Wellness_Vitamins": 0.9, "Fitness_Weights": 0.6}, "payout": 25.00},
            "Insomnia Cure Sleep Tracker": {"vector": {"Wellness_Sleep": 0.9, "Tech_Smartwatch": 0.8}, "payout": 60.00},
            "Adrenaline-Pumping Action Blockbuster": {"vector": {"Cinema_Action": 0.9, "FastFood_Pizza": 0.6}, "payout": 15.00},
            "Feel-Good Romantic Comedy": {"vector": {"Cinema_Comedy": 0.9, "FineDining_Wine": 0.6}, "payout": 10.00},
            "Mind-Bending SciFi Epic": {"vector": {"Cinema_SciFi": 0.9, "Tech_Laptop": 0.5}, "payout": 15.00},
            "Heart-Pounding Horror Film": {"vector": {"Cinema_Horror": 0.9, "FastFood_Burger": 0.5}, "payout": 12.00},
            "Tear-Jerker Prestige Drama": {"vector": {"Cinema_Drama": 0.9, "CoffeeShop_Tea": 0.7}, "payout": 10.00},
            "Upbeat Pop Motivation Playlist": {"vector": {"Music_Pop": 0.9, "Fitness_Cardio": 0.7}, "payout": 5.00},
            "Rage-Release EDM Festival Ticket": {"vector": {"Music_EDM": 0.9, "Beverages_EnergyDrink": 0.6}, "payout": 150.00},
            "Confidence-Boost HipHop Album": {"vector": {"Music_HipHop": 0.9, "Fashion_Streetwear": 0.8}, "payout": 15.00},
            "Classic Rock Nostalgia Tour": {"vector": {"Music_Rock": 0.9, "Cars_SportsCar": 0.5}, "payout": 85.00},
            "Focus-Enhancing Classical Stream": {"vector": {"Music_Classical": 0.9, "Education_Math": 0.8}, "payout": 8.00},
            "Escapist Fantasy Novel Box Set": {"vector": {"Books_Fantasy": 0.9, "CoffeeShop_Tea": 0.6}, "payout": 40.00},
            "Inspiring Tech Biography": {"vector": {"Books_Biography": 0.9, "Tech_Smartphone": 0.7}, "payout": 20.00},
            "Nail-Biting Crime Thriller": {"vector": {"Books_Thriller": 0.9, "Cinema_Drama": 0.6}, "payout": 15.00},
            "Cozy Romance Paperback": {"vector": {"Books_Romance": 0.9, "Wellness_Massage": 0.5}, "payout": 12.00},
            "Life-Changing Self Help Audio": {"vector": {"Books_SelfHelp": 0.9, "Wellness_Meditation": 0.7}, "payout": 18.00},
            "Immersive RPG 100-Hour Escape": {"vector": {"Gaming_RPG": 0.9, "Beverages_Soda": 0.7}, "payout": 60.00},
            "Aggressive FPS Shooter Pre-Order": {"vector": {"Gaming_FPS": 0.9, "Beverages_EnergyDrink": 0.8}, "payout": 70.00},
            "Brain-Teasing Strategy Simulator": {"vector": {"Gaming_Strategy": 0.9, "Education_Coding": 0.6}, "payout": 40.00},
            "Competitive eSports Pass": {"vector": {"Gaming_Sports": 0.9, "Sports_Soccer": 0.7}, "payout": 25.00},
            "Relaxing Indie Puzzle Game": {"vector": {"Gaming_Puzzle": 0.9, "CoffeeShop_Latte": 0.6}, "payout": 15.00},
            "Binge-Worthy Netflix Crime Series": {"vector": {"Streaming_Netflix": 0.9, "FastFood_Pizza": 0.7}, "payout": 15.00},
            "Comfort Hulu Sitcom Marathon": {"vector": {"Streaming_Hulu": 0.9, "Home_Furniture": 0.6}, "payout": 12.00},
            "Nostalgic Disney+ Movie Night": {"vector": {"Streaming_Disney": 0.9, "Pets_Dog": 0.4}, "payout": 10.00},
            "Prestige HBO Drama Subscription": {"vector": {"Streaming_HBO": 0.9, "FineDining_Wine": 0.7}, "payout": 18.00},
            "Weekend Prime Video Binge": {"vector": {"Streaming_Prime": 0.9, "FastFood_Burger": 0.6}, "payout": 14.00},
            "Anger-Management Heavy Kettlebells": {"vector": {"Fitness_Weights": 0.9, "Beverages_Water": 0.6}, "payout": 55.00},
            "Endurance-Testing Marathon Shoes": {"vector": {"Fitness_Cardio": 0.9, "Fashion_Activewear": 0.8}, "payout": 130.00},
            "Inner-Peace Premium Yoga Mat": {"vector": {"Fitness_Yoga": 0.9, "Wellness_Meditation": 0.7}, "payout": 35.00},
            "Refreshing Olympic Swim Goggles": {"vector": {"Fitness_Swimming": 0.9, "Travel_Beach": 0.5}, "payout": 25.00},
            "Stress-Relief Heavy Punching Bag": {"vector": {"Fitness_Boxing": 0.9, "Gaming_FPS": 0.4}, "payout": 90.00},
            "Tropical Beach Getaway Package": {"vector": {"Travel_Beach": 0.9, "Fashion_Casual": 0.6}, "payout": 800.00},
            "Isolated Mountain Cabin Retreat": {"vector": {"Travel_Mountain": 0.9, "Books_Thriller": 0.5}, "payout": 450.00},
            "Bustling City Adventure Flight": {"vector": {"Travel_City": 0.9, "FineDining_Sushi": 0.6}, "payout": 300.00},
            "Luxury Ocean Cruise All-Inclusive": {"vector": {"Travel_Cruise": 0.9, "FineDining_Seafood": 0.7}, "payout": 1200.00},
            "Off-Grid Survival Camping Gear": {"vector": {"Travel_Camping": 0.9, "Hobbies_DIY": 0.6}, "payout": 150.00},
            "Team-Spirit Soccer Cleats": {"vector": {"Sports_Soccer": 0.9, "Beverages_Water": 0.5}, "payout": 120.00},
            "High-Energy Basketball Hoops": {"vector": {"Sports_Basketball": 0.9, "Fashion_Streetwear": 0.6}, "payout": 85.00},
            "Focused Pro Tennis Racket": {"vector": {"Sports_Tennis": 0.9, "Fashion_Activewear": 0.7}, "payout": 160.00},
            "Relaxed Afternoon Baseball Cap": {"vector": {"Sports_Baseball": 0.9, "Beverages_Beer": 0.6}, "payout": 30.00},
            "Calm Weekend Golf Club Set": {"vector": {"Sports_Golf": 0.9, "Cars_SUV": 0.5}, "payout": 400.00},
            "Loyal Dog Training & Care Course": {"vector": {"Pets_Dog": 0.9, "Travel_Mountain": 0.4}, "payout": 45.00},
            "Cozy Cat Tree & Scratching Post": {"vector": {"Pets_Cat": 0.9, "Home_Furniture": 0.6}, "payout": 65.00},
            "Cheerful Bird Cage Setup": {"vector": {"Pets_Bird": 0.9, "Home_Decor": 0.5}, "payout": 80.00},
            "Calming Zen Aquarium Tank": {"vector": {"Pets_Fish": 0.9, "Wellness_Meditation": 0.5}, "payout": 150.00},
            "Exotic Reptile Heat Lamp": {"vector": {"Pets_Reptile": 0.9, "Tech_Smartwatch": 0.2}, "payout": 40.00},
            "Mindful DSLR Camera Lens": {"vector": {"Hobbies_Photography": 0.9, "Travel_City": 0.6}, "payout": 350.00},
            "Creative Flow Canvas & Paints": {"vector": {"Hobbies_Painting": 0.9, "Music_Classical": 0.5}, "payout": 45.00},
            "Therapeutic Backyard Greenhouse": {"vector": {"Hobbies_Gardening": 0.9, "Home_Garden": 0.8}, "payout": 250.00},
            "Comforting Masterchef Knife Set": {"vector": {"Hobbies_Cooking": 0.9, "FineDining_Steak": 0.6}, "payout": 110.00},
            "Satisfying DIY Power Drill": {"vector": {"Hobbies_DIY": 0.9, "Home_Appliances": 0.6}, "payout": 95.00},
            "Status-Symbol Pro Smartphone": {"vector": {"Tech_Smartphone": 0.9, "Fashion_Streetwear": 0.6}, "payout": 999.00},
            "Productivity-Boosting Ultrabook": {"vector": {"Tech_Laptop": 0.9, "Education_Coding": 0.7}, "payout": 1200.00},
            "Entertainment OLED Tablet": {"vector": {"Tech_Tablet": 0.9, "Streaming_Netflix": 0.8}, "payout": 500.00},
            "Fitness-Tracking Smartwatch": {"vector": {"Tech_Smartwatch": 0.9, "Fitness_Cardio": 0.8}, "payout": 250.00},
            "Isolation Noise-Canceling Headphones": {"vector": {"Tech_Headphones": 0.9, "Travel_City": 0.6}, "payout": 300.00},
            "Confidence-Boosting Hypebeast Shoes": {"vector": {"Fashion_Streetwear": 0.9, "Music_HipHop": 0.7}, "payout": 220.00},
            "Power-Trip Formal Tailored Suit": {"vector": {"Fashion_Formal": 0.9, "FineDining_Steak": 0.6}, "payout": 400.00},
            "Cozy Sunday Casual Lounge Set": {"vector": {"Fashion_Casual": 0.9, "Streaming_Hulu": 0.7}, "payout": 65.00},
            "Motivation High-Performance Activewear": {"vector": {"Fashion_Activewear": 0.9, "Fitness_Weights": 0.7}, "payout": 90.00},
            "Unique Vintage Thrift Jacket": {"vector": {"Fashion_Vintage": 0.9, "Hobbies_Photography": 0.5}, "payout": 55.00},
            "Thrill-Seeking Sports Car Lease": {"vector": {"Cars_SportsCar": 0.9, "Fashion_Formal": 0.5}, "payout": 800.00},
            "Family-Safe Suburban SUV": {"vector": {"Cars_SUV": 0.9, "Pets_Dog": 0.5}, "payout": 600.00},
            "Reliable Commuter Sedan": {"vector": {"Cars_Sedan": 0.9, "CoffeeShop_Espresso": 0.4}, "payout": 300.00},
            "Rugged Off-Road 4x4 Truck": {"vector": {"Cars_Truck": 0.9, "Travel_Camping": 0.7}, "payout": 700.00},
            "Eco-Guilt-Free EV Charger Install": {"vector": {"Cars_EV": 0.9, "Tech_Smartphone": 0.5}, "payout": 450.00},
            "Comforting Plush Sectional Sofa": {"vector": {"Home_Furniture": 0.9, "Streaming_Netflix": 0.6}, "payout": 850.00},
            "Mood-Setting Smart LED Bulbs": {"vector": {"Home_Lighting": 0.9, "Gaming_FPS": 0.5}, "payout": 60.00},
            "Aesthetic Minimalist Room Decor": {"vector": {"Home_Decor": 0.9, "Hobbies_Photography": 0.5}, "payout": 85.00},
            "Life-Simplifying Robot Vacuum": {"vector": {"Home_Appliances": 0.9, "Pets_Cat": 0.6}, "payout": 350.00},
            "Peaceful Zen Garden Water Feature": {"vector": {"Home_Garden": 0.9, "Wellness_Meditation": 0.6}, "payout": 120.00},
            "Cultural Escape Language App": {"vector": {"Education_Languages": 0.9, "Travel_City": 0.6}, "payout": 30.00},
            "Career-Pivoting Python Bootcamp": {"vector": {"Education_Coding": 0.9, "Tech_Laptop": 0.7}, "payout": 150.00},
            "Curiosity-Satisfying History Docuseries": {"vector": {"Education_History": 0.9, "Books_Biography": 0.6}, "payout": 20.00},
            "Mind-Expanding Quantum Physics Course": {"vector": {"Education_Science": 0.9, "Cinema_SciFi": 0.5}, "payout": 45.00},
            "Logic-Building Advanced Math Tutor": {"vector": {"Education_Math": 0.9, "Gaming_Strategy": 0.5}, "payout": 80.00}
        }
        
        evaluation_results = []
        
        for ad_name, details in ad_inventory.items():
            p_short = calculate_cosine_similarity(self.short_term, details["vector"])
            p_long = calculate_cosine_similarity(self.long_term, details["vector"])
            
            combined_probability = (self.weight_short * p_short) + (self.weight_long * p_long)
            if combined_probability <= 0.10: # The 10% Relevance Gate
                continue

            expected_utility = combined_probability * details["payout"]

            
            
            # Only track items that actually have a mathematical probability
            if expected_utility > 0:
                evaluation_results.append({
                    "name": ad_name,
                    "prob": combined_probability,
                    "payout": details["payout"],
                    "eu": expected_utility
                })
            
        evaluation_results.sort(key=lambda x: x["eu"], reverse=True)
        
        print("\n>>> DECISION ENGINE: Calculating Expected Utility <<<")
        
        if not evaluation_results:
            print("  [Waiting for user data to calculate utility...]")
            print(f"{'='*55}\n")
            return

        print("  [Displaying Top 5 Competitors]:")
        for rank, item in enumerate(evaluation_results[:5], 1):
             print(f"    #{rank}. [{item['name']}]")
             print(f"        P({item['prob']:.2f}) * Payout(${item['payout']:.2f}) = EU: ${item['eu']:.2f}")
                
        best_ad = evaluation_results[0]
        
        print(f"\n[!] INFLUENCE DECISION: Serving Ad -> [{best_ad['name']}]")
        print(f"    (Maximizes Expected Profit at ${best_ad['eu']:.2f})")
        print(f"{'='*55}\n")

# =====================================================================
# 4. MAIN SIMULATION LOOP
# =====================================================================
if _name_ == "_main_":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please paste your Gemini API key at the top of the file!")
        exit()

    print("Booting Enterprise AI Recommendation Engine...\n")
    time.sleep(1)
    
    # --- BUILD THE BLANK 100-ITEM MATRIX (0.0 Baseline) ---
    starting_vector = {
        "CoffeeShop_Espresso": 0.0, "CoffeeShop_Latte": 0.0, "CoffeeShop_ColdBrew": 0.0, "CoffeeShop_Pastry": 0.0, "CoffeeShop_Tea": 0.0,
        "FastFood_Burger": 0.0, "FastFood_Pizza": 0.0, "FastFood_Fries": 0.0, "FastFood_Tacos": 0.0, "FastFood_Chicken": 0.0,
        "FineDining_Steak": 0.0, "FineDining_Sushi": 0.0, "FineDining_Wine": 0.0, "FineDining_Pasta": 0.0, "FineDining_Seafood": 0.0,
        "Beverages_Water": 0.0, "Beverages_Soda": 0.0, "Beverages_Juice": 0.0, "Beverages_EnergyDrink": 0.0, "Beverages_Beer": 0.0,
        "Wellness_Meditation": 0.0, "Wellness_Skincare": 0.0, "Wellness_Massage": 0.0, "Wellness_Vitamins": 0.0, "Wellness_Sleep": 0.0,
        "Cinema_Action": 0.0, "Cinema_Comedy": 0.0, "Cinema_SciFi": 0.0, "Cinema_Horror": 0.0, "Cinema_Drama": 0.0,
        "Music_Pop": 0.0, "Music_EDM": 0.0, "Music_HipHop": 0.0, "Music_Rock": 0.0, "Music_Classical": 0.0,
        "Books_Fantasy": 0.0, "Books_Biography": 0.0, "Books_Thriller": 0.0, "Books_Romance": 0.0, "Books_SelfHelp": 0.0,
        "Gaming_RPG": 0.0, "Gaming_FPS": 0.0, "Gaming_Strategy": 0.0, "Gaming_Sports": 0.0, "Gaming_Puzzle": 0.0,
        "Streaming_Netflix": 0.0, "Streaming_Hulu": 0.0, "Streaming_Disney": 0.0, "Streaming_HBO": 0.0, "Streaming_Prime": 0.0,
        "Fitness_Weights": 0.0, "Fitness_Cardio": 0.0, "Fitness_Yoga": 0.0, "Fitness_Swimming": 0.0, "Fitness_Boxing": 0.0,
        "Travel_Beach": 0.0, "Travel_Mountain": 0.0, "Travel_City": 0.0, "Travel_Cruise": 0.0, "Travel_Camping": 0.0,
        "Sports_Soccer": 0.0, "Sports_Basketball": 0.0, "Sports_Tennis": 0.0, "Sports_Baseball": 0.0, "Sports_Golf": 0.0,
        "Pets_Dog": 0.0, "Pets_Cat": 0.0, "Pets_Bird": 0.0, "Pets_Fish": 0.0, "Pets_Reptile": 0.0,
        "Hobbies_Photography": 0.0, "Hobbies_Painting": 0.0, "Hobbies_Gardening": 0.0, "Hobbies_Cooking": 0.0, "Hobbies_DIY": 0.0,
        "Tech_Smartphone": 0.0, "Tech_Laptop": 0.0, "Tech_Tablet": 0.0, "Tech_Smartwatch": 0.0, "Tech_Headphones": 0.0,
        "Fashion_Streetwear": 0.0, "Fashion_Formal": 0.0, "Fashion_Casual": 0.0, "Fashion_Activewear": 0.0, "Fashion_Vintage": 0.0,
        "Cars_SportsCar": 0.0, "Cars_SUV": 0.0, "Cars_Sedan": 0.0, "Cars_Truck": 0.0, "Cars_EV": 0.0,
        "Home_Furniture": 0.0, "Home_Lighting": 0.0, "Home_Decor": 0.0, "Home_Appliances": 0.0, "Home_Garden": 0.0,
        "Education_Languages": 0.0, "Education_Coding": 0.0, "Education_History": 0.0, "Education_Science": 0.0, "Education_Math": 0.0
    }
    
    # We removed the Gamer Anchors. The user is a blank slate.
    student = ContextAwareUserModel(user_id="Session_409", initial_long_term_stats=starting_vector)
    
    student.print_state()
    student.predict_character()
    student.serve_best_ad()
    
    print("\nINSTRUCTIONS FOR DEMO:")
    print("- You are a blank slate. Type a sentence to create your personality.")
    print("- Type 'exit' to quit.")
    print("-" * 55)

    valid_matrix_keys = list(student.long_term.keys())

    while True:
        user_input = input("\n[USER INPUT]: ").strip()
        
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue
            
        print("  -> NLP Engine is analyzing emotional intent...")
        semantic_impact = parse_with_real_ai(user_input, valid_matrix_keys)
        
        if not semantic_impact:
            print("[System]: No strong matrix correlation detected. Try again.")
            continue
            
        print(f"\n[Extracted Matrix Weights: {semantic_impact}]")
        
        student.register_action(semantic_impact)
        
        student.print_state()
        student.predict_character()
        student.serve_best_ad()