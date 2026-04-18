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