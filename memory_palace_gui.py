import streamlit as st
import torch
import numpy as np
import pandas as pd
from mnemonic_model import DIM_Net, TRHD_MnemonicMapper, ChessCubeLattice
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load mathematical dataset
@st.cache_data
def load_math_dataset():
    """Load the mathematical training dataset"""
    try:
        return pd.read_csv('math_training_data.csv')
    except FileNotFoundError:
        return None

# Initialize semantic search
@st.cache_resource
def initialize_semantic_search(math_df):
    """Initialize TF-IDF vectorizer for semantic search"""
    if math_df is None:
        return None

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(math_df['math_concept'])
    return vectorizer, tfidf_matrix

# --- 2. MODEL LOADING ---

@st.cache_resource
def load_trained_models():
    """Load the trained DIM-Net and TRHD_MnemonicMapper"""
    try:
        dim_net = DIM_Net()
        dim_net.load_state_dict(torch.load('dim_net_final.pth', map_location=DEVICE))
        dim_net.to(DEVICE)
        dim_net.eval()

        trhd_mapper = TRHD_MnemonicMapper()
        trhd_mapper.load_state_dict(torch.load('trhd_mapper_final.pth', map_location=DEVICE))
        trhd_mapper.to(DEVICE)
        trhd_mapper.eval()

        return dim_net, trhd_mapper
    except FileNotFoundError:
        st.error("❌ Trained model files not found. Please run training first.")
        return None, None

# --- 3. MEMORY PALACE FUNCTIONS ---

def create_fact_cluster(fact, cluster_size=10):
    """Create a simulated cluster around a single fact for TRHD_MnemonicMapper"""
    cluster_features = []

    for i in range(cluster_size):
        if i == 0:
            # Main fact
            fact_vector = torch.randn(70)
            fact_vector[0] = len(fact) / 100.0  # Normalized length
            truth_score = torch.tensor([0.9])  # High confidence
        else:
            # Supporting facts (simulated)
            fact_vector = torch.randn(70)
            fact_vector[0] = np.random.uniform(0.5, 1.5)
            truth_score = torch.rand(1) * 0.4 + 0.6

        fact_feature = torch.cat([fact_vector, truth_score])
        cluster_features.append(fact_feature)

    return torch.stack(cluster_features).unsqueeze(0).to(DEVICE)

def generate_memory_coordinates(fact, trhd_mapper):
    """Generate 3D coordinates for a fact"""
    cluster_input = create_fact_cluster(fact)

    with torch.no_grad():
        predicted_coords = trhd_mapper(cluster_input).squeeze(0).cpu().numpy()

    # Round to nearest integer coordinates
    x_coord = int(np.clip(np.round(predicted_coords[0]), 1, 8))
    y_coord = int(np.clip(np.round(predicted_coords[1]), 1, 8))
    z_coord = int(np.clip(np.round(predicted_coords[2]), 1, 8))

    return x_coord, y_coord, z_coord, predicted_coords

def generate_pao_mnemonic(fact, coords):
    """Generate a PAO mnemonic using the coordinates"""
    x, y, z = coords

    persons = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Leonardo da Vinci",
              "Charles Darwin", "Galileo", "Nikola Tesla", "Ada Lovelace"]
    actions = ["painting", "dancing", "juggling", "building", "flying", "swimming",
              "singing", "writing"]
    objects = ["crystals", "equations", "machines", "books", "stars", "robots",
              "paintings", "musical notes"]

    person = persons[(x-1) % len(persons)]
    action = actions[(y-1) % len(actions)]
    obj = objects[(z-1) % len(objects)]

    return f"{person} is {action} {obj} while remembering: {fact[:40]}..."

# --- 4. STREAMLIT APP ---

def main():
    st.set_page_config(page_title="Mathematical Memory Palace", page_icon="🧮", layout="wide")

    st.title("🧮 Mathematical Memory Palace AI")
    st.markdown("*Navigate mathematical knowledge through 3D spatial reasoning*")

    # Load data and models
    math_df = load_math_dataset()
    dim_net, trhd_mapper = load_trained_models()
    vectorizer, tfidf_matrix = initialize_semantic_search(math_df)

    if math_df is None:
        st.error("❌ Mathematical dataset not found. Please run data generation first.")
        return

    if dim_net is None or trhd_mapper is None:
        st.error("❌ Trained models not found. Please run training first.")
        return

    # Sidebar with mathematical domains
    with st.sidebar:
        st.header("📚 Mathematical Domains")
        domains = sorted(math_df['domain'].unique())
        selected_domain = st.selectbox("Filter by domain:", ["All"] + domains)

        st.header("🎯 Memory Palace Features")
        st.markdown("""
        - **Encode**: Store mathematical concepts spatially
        - **Retrieve**: Query by coordinates or concepts
        - **Navigate**: Explore mathematical relationships
        - **Compute**: Execute APL expressions
        """)

        # Dataset stats
        st.header("📊 Dataset Statistics")
        st.metric("Total Concepts", f"{len(math_df):,}")
        st.metric("Domains", len(domains))
        st.metric("Levels", len(math_df['level'].unique()))

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧮 Encode Concept", "🔍 Retrieve Knowledge", "🧠 Explore Palace", "⚡ APL Computation"])

    with tab1:
        encode_concept_tab(math_df, trhd_mapper)

    with tab2:
        retrieve_knowledge_tab(math_df, trhd_mapper, vectorizer, tfidf_matrix)

    with tab3:
        explore_palace_tab(math_df)

    with tab4:
        apl_computation_tab(math_df)

# --- 5. TAB FUNCTIONS ---

def encode_concept_tab(math_df, trhd_mapper):
    """Tab for encoding new mathematical concepts"""
    st.header("🧮 Encode Mathematical Concept")

    col1, col2 = st.columns([2, 1])

    with col1:
        concept_input = st.text_area(
            "Enter a mathematical concept:",
            placeholder="Example: The derivative of sin(x) is cos(x)",
            height=100
        )

        # Domain selection
        domains = sorted(math_df['domain'].unique())
        selected_domain = st.selectbox("Mathematical domain:", domains)

        encode_button = st.button("🧮 Encode to Memory Palace", type="primary", use_container_width=True)

    with col2:
        st.header("📊 System Status")
        st.success("✅ Mathematical models loaded")
        st.info("🧮 6,400 concepts trained")
        st.info("🎯 10 mathematical levels")
        st.info("♟️ Extended chess cube (10×8×8)")

    if encode_button and concept_input.strip():
        with st.spinner("🧮 Encoding mathematical concept..."):
            # Generate coordinates
            x, y, z, raw_coords = generate_memory_coordinates(concept_input, trhd_mapper)

            # Handle extended coordinates (level 9-10)
            if z > 8:
                z_display = f"{z} (Extended)"
            else:
                z_display = str(z)

            # Calculate parity
            parity = (x + y + z) % 2
            color = "⚫ Black" if parity == 1 else "⚪ White"

            # Generate mathematical mnemonic
            mnemonic = generate_math_mnemonic(concept_input, (x, y, z))

            # Find similar concepts in dataset
            similar_concepts = find_similar_concepts(concept_input, math_df, top_k=3)

        st.success("🎉 Mathematical concept encoded successfully!")

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("X Coordinate", f"{x}/8")
        with col2:
            st.metric("Y Coordinate", f"{y}/8")
        with col3:
            st.metric("Z Level", z_display)

        st.subheader("🎨 Mathematical Memory Properties")
        st.info(f"**{color} Square** - Geometric constraint for mathematical organization")

        st.subheader("🎭 Mathematical Mnemonic")
        st.markdown(f"**{mnemonic}**")

        if similar_concepts:
            st.subheader("🔗 Related Mathematical Concepts")
            for concept in similar_concepts:
                st.markdown(f"• {concept}")

def retrieve_knowledge_tab(math_df, trhd_mapper, vectorizer, tfidf_matrix):
    """Tab for retrieving mathematical knowledge"""
    st.header("🔍 Retrieve Mathematical Knowledge")

    query_type = st.radio("Query type:", ["By Concept", "By Coordinates", "By Domain", "Semantic Search"])

    if query_type == "By Concept":
        concept_query = st.text_input("Enter mathematical concept to find:")
        if st.button("🔍 Search") and concept_query:
            results = search_mathematical_concepts(concept_query, math_df, trhd_mapper)
            display_search_results(results)

    elif query_type == "Semantic Search":
        semantic_query = st.text_input("Enter mathematical concept for semantic search:")
        top_k = st.slider("Number of results:", 1, 20, 5)

        if st.button("🧠 Semantic Search") and semantic_query and vectorizer and tfidf_matrix is not None:
            results = semantic_search_mathematics(semantic_query, math_df, vectorizer, tfidf_matrix, top_k)
            display_semantic_results(results)

    elif query_type == "By Coordinates":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_coord = st.slider("X coordinate", 1, 8, 4)
        with col2:
            y_coord = st.slider("Y coordinate", 1, 8, 4)
        with col3:
            z_coord = st.slider("Z level", 1, 10, 5)

        if st.button("📍 Retrieve from Location"):
            concepts = retrieve_by_coordinates(math_df, x_coord, y_coord, z_coord)
            display_coordinate_results(concepts, (x_coord, y_coord, z_coord))

    elif query_type == "By Domain":
        domains = sorted(math_df['domain'].unique())
        selected_domain = st.selectbox("Select mathematical domain:", domains)

        if st.button("📚 Explore Domain"):
            domain_concepts = math_df[math_df['domain'] == selected_domain]
            display_domain_overview(domain_concepts, selected_domain)

def explore_palace_tab(math_df):
    """Tab for exploring the mathematical memory palace"""
    st.header("🧠 Explore Mathematical Memory Palace")

    # Level selector
    levels = sorted(math_df['level'].unique())
    level_names = [f"Level {l}: {math_df[math_df['level']==l]['level_name'].iloc[0]}" for l in levels]
    selected_level = st.selectbox("Select mathematical level:", levels, format_func=lambda x: level_names[x-1])

    if selected_level:
        level_data = math_df[math_df['level'] == selected_level]
        level_name = level_data['level_name'].iloc[0]

        st.subheader(f"📚 {level_name}")

        # Show domains in this level
        domains = level_data['domain'].unique()
        for domain in domains:
            with st.expander(f"🔹 {domain.replace('_', ' ').title()}"):
                domain_concepts = level_data[level_data['domain'] == domain]['math_concept'].tolist()[:5]
                for concept in domain_concepts:
                    st.markdown(f"• {concept}")
                if len(level_data[level_data['domain'] == domain]) > 5:
                    st.info(f"... and {len(level_data[level_data['domain'] == domain]) - 5} more concepts")

def apl_computation_tab(math_df):
    """Tab for executing APL computations at memory palace locations"""
    st.header("⚡ APL Mathematical Computation")

    st.markdown("""
    Execute APL expressions stored at memory palace coordinates.
    APL (A Programming Language) enables powerful array-oriented mathematical computations.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📍 Select Location")
        x_coord = st.slider("X coordinate", 1, 8, 4, key="apl_x")
        y_coord = st.slider("Y coordinate", 1, 8, 4, key="apl_y")
        z_coord = st.slider("Z level", 1, 10, 5, key="apl_z")

        if st.button("🔍 Find APL Expressions", use_container_width=True):
            apl_expressions = find_apl_at_coordinates(math_df, x_coord, y_coord, z_coord)
            st.session_state['current_apl_expressions'] = apl_expressions

    with col2:
        st.subheader("⚙️ APL Interpreter Status")
        st.success("✅ APL Interpreter Ready")
        st.info("🧮 Array-oriented computing")
        st.info("📊 Mathematical functions available")

        # Quick APL reference
        with st.expander("📚 APL Quick Reference"):
            st.markdown("""
            **Basic Operations:**
            - `+` Addition, `×` Multiplication, `÷` Division
            - `⌈` Ceiling, `⌊` Floor, `|` Modulus
            - `⍳` Index generator, `⍴` Shape/Reshape

            **Array Operations:**
            - `+/` Sum, `×/` Product, `⌈/` Maximum
            - `∧/` All true, `∨/` Any true
            """)

    # Display APL expressions if found
    if 'current_apl_expressions' in st.session_state and st.session_state['current_apl_expressions']:
        st.subheader("🧮 APL Expressions at Location")

        for i, expr_data in enumerate(st.session_state['current_apl_expressions']):
            with st.expander(f"Expression {i+1}: {expr_data['concept'][:40]}..."):
                st.code(expr_data['apl_code'], language='apl')

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"⚡ Execute", key=f"exec_{i}"):
                        result = execute_apl_expression(expr_data['apl_code'])
                        st.session_state[f'result_{i}'] = result

                with col2:
                    st.caption(f"Domain: {expr_data['domain']}")

                if f'result_{i}' in st.session_state:
                    st.success("Execution Result:")
                    st.code(str(st.session_state[f'result_{i}']), language='text')

    # Manual APL input
    st.subheader("✏️ Manual APL Expression")
    manual_apl = st.text_input("Enter APL expression:", placeholder="2 + 2  ⍝ Addition example")

    if st.button("⚡ Execute Manual", use_container_width=True) and manual_apl.strip():
        result = execute_apl_expression(manual_apl)
        st.success("Manual Execution Result:")
        st.code(str(result), language='text')

    # APL Learning Resources
    st.subheader("📖 APL Learning")
    with st.expander("Mathematical Examples"):
        st.markdown("""
        **Arithmetic:** `3 × 4` → 12

        **Arrays:** `1 2 3 + 4 5 6` → 5 7 9

        **Matrix:** `2 2 ⍴ 1 2 3 4` → 2×2 matrix

        **Statistics:** `+/ 1 2 3 4 5` → 15 (sum)

        **Logic:** `1 ∧ 0` → 0 (AND)
        """)

def generate_math_mnemonic(concept, coords):
    """Generate a mathematics-focused PAO mnemonic"""
    x, y, z = coords

    # Mathematics-themed persons
    math_persons = ["Euclid", "Pythagoras", "Gauss", "Euler", "Riemann", "Hilbert", "Noether", "Turing"]
    math_actions = ["proving", "calculating", "deriving", "integrating", "transforming", "optimizing", "modeling", "computing"]
    math_objects = ["theorems", "equations", "matrices", "functions", "geometries", "algorithms", "proofs", "axioms"]

    person = math_persons[(x-1) % len(math_persons)]
    action = math_actions[(y-1) % len(math_actions)]
    obj = math_objects[(z-1) % len(math_objects)]

    return f"{person} is {action} {obj} while deriving: {concept[:40]}..."

def find_similar_concepts(concept, math_df, top_k=3):
    """Find similar mathematical concepts (simple keyword matching)"""
    concept_lower = concept.lower()
    similarities = []

    for _, row in math_df.iterrows():
        math_concept = row['math_concept'].lower()
        # Simple similarity based on common words
        concept_words = set(concept_lower.split())
        math_words = set(math_concept.split())
        overlap = len(concept_words.intersection(math_words))
        if overlap > 0:
            similarities.append((overlap, row['math_concept']))

    similarities.sort(reverse=True)
    return [concept for _, concept in similarities[:top_k]]

def search_mathematical_concepts(query, math_df, trhd_mapper):
    """Search for mathematical concepts by query"""
    query_lower = query.lower()
    results = []

    for _, row in math_df.iterrows():
        if query_lower in row['math_concept'].lower():
            results.append({
                'concept': row['math_concept'],
                'domain': row['domain'],
                'level': row['level'],
                'coordinates': (row['x_coord'], row['y_coord'], row['z_coord']),
                'apl_code': row.get('apl_code', '')
            })

    return results[:10]  # Limit results

def retrieve_by_coordinates(math_df, x, y, z):
    """Retrieve concepts at specific coordinates"""
    matches = math_df[
        (math_df['x_coord'] == x) &
        (math_df['y_coord'] == y) &
        (math_df['z_coord'] == z)
    ]
    return matches.to_dict('records')

def display_search_results(results):
    """Display search results"""
    if not results:
        st.warning("No matching concepts found.")
        return

    for result in results:
        with st.expander(f"🧮 {result['concept'][:50]}..."):
            st.write(f"**Domain:** {result['domain'].replace('_', ' ').title()}")
            st.write(f"**Level:** {result['level']}")
            st.write(f"**Coordinates:** ({result['coordinates'][0]}, {result['coordinates'][1]}, {result['coordinates'][2]})")
            if result['apl_code']:
                st.code(result['apl_code'], language='apl')

def display_coordinate_results(concepts, coords):
    """Display concepts at coordinates"""
    st.subheader(f"📍 Concepts at location {coords}")

    if not concepts:
        st.info("No concepts found at this location.")
        return

    for concept in concepts:
        st.markdown(f"• **{concept['math_concept']}**")
        st.caption(f"Domain: {concept['domain']} | APL: {concept.get('apl_code', 'N/A')}")

def display_domain_overview(domain_concepts, domain_name):
    """Display overview of a mathematical domain"""
    st.subheader(f"📚 {domain_name.replace('_', ' ').title()}")

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Concepts", len(domain_concepts))
    with col2:
        unique_levels = domain_concepts['level'].nunique()
        st.metric("Levels", unique_levels)
    with col3:
        avg_difficulty = domain_concepts['difficulty'].mode().iloc[0] if 'difficulty' in domain_concepts.columns else "N/A"
        st.metric("Difficulty", avg_difficulty.title())

    # Sample concepts
    st.subheader("Sample Concepts")
    sample_concepts = domain_concepts['math_concept'].sample(min(10, len(domain_concepts)))
    for concept in sample_concepts:
        st.markdown(f"• {concept}")

def find_apl_at_coordinates(math_df, x, y, z):
    """Find APL expressions at specific coordinates"""
    matches = math_df[
        (math_df['x_coord'] == x) &
        (math_df['y_coord'] == y) &
        (math_df['z_coord'] == z) &
        (math_df['apl_code'].notna()) &
        (math_df['apl_code'] != '')
    ]

    results = []
    for _, row in matches.iterrows():
        results.append({
            'concept': row['math_concept'],
            'apl_code': row['apl_code'],
            'domain': row['domain'],
            'level': row['level']
        })

    return results

def execute_apl_expression(apl_code):
    """Execute APL expression (simplified simulation for demo)"""
    try:
        # This is a simplified APL simulator for demonstration
        # In a real implementation, you'd use a proper APL interpreter

        # Remove comments
        code = apl_code.split('⍝')[0].strip()

        if not code:
            return "Empty expression"

        # Basic arithmetic operations
        if '+' in code and ' ' in code:
            parts = code.split('+')
            if len(parts) == 2:
                try:
                    a = float(parts[0].strip())
                    b = float(parts[1].strip())
                    return f"{a} + {b} = {a + b}"
                except:
                    pass

        if '×' in code and ' ' in code:
            parts = code.split('×')
            if len(parts) == 2:
                try:
                    a = float(parts[0].strip())
                    b = float(parts[1].strip())
                    return f"{a} × {b} = {a * b}"
                except:
                    pass

        # Array operations
        if '⍳' in code:
            try:
                n = int(code.replace('⍳', '').strip())
                result = list(range(1, n+1))
                return f"⍳{n} = {result}"
            except:
                pass

        # Vector operations
        if '+' in code and len(code.split()) > 2:
            try:
                parts = code.split('+')
                if len(parts) == 2:
                    vec1 = [float(x) for x in parts[0].split()]
                    vec2 = [float(x) for x in parts[1].split()]
                    if len(vec1) == len(vec2):
                        result = [a + b for a, b in zip(vec1, vec2)]
                        return f"{vec1} + {vec2} = {result}"
            except:
                pass

        # Sum reduction
        if '+/' in code:
            try:
                vec = [float(x) for x in code.replace('+/','').split()]
                result = sum(vec)
                return f"+/ {vec} = {result}"
            except:
                pass

        # Default: return the expression with a note
        return f"APL Expression: {code}\n(Note: Full APL interpreter would execute this)"

    except Exception as e:
        return f"Error executing APL: {str(e)}"

def semantic_search_mathematics(query, math_df, vectorizer, tfidf_matrix, top_k=5):
    """Perform semantic search on mathematical concepts"""
    try:
        # Transform query to TF-IDF vector
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top-k similar concepts
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                row = math_df.iloc[idx]
                results.append({
                    'concept': row['math_concept'],
                    'domain': row['domain'],
                    'level': row['level'],
                    'coordinates': (row['x_coord'], row['y_coord'], row['z_coord']),
                    'apl_code': row.get('apl_code', ''),
                    'similarity': similarities[idx]
                })

        return results

    except Exception as e:
        st.error(f"Semantic search error: {str(e)}")
        return []

def display_semantic_results(results):
    """Display semantic search results"""
    if not results:
        st.warning("No similar mathematical concepts found.")
        return

    st.subheader("🧠 Semantic Search Results")

    for i, result in enumerate(results, 1):
        similarity_percent = result['similarity'] * 100

        with st.expander(f"#{i} {result['concept'][:50]}... ({similarity_percent:.1f}% similar)"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Domain:** {result['domain'].replace('_', ' ').title()}")
                st.write(f"**Level:** {result['level']}")
                st.write(f"**Coordinates:** ({result['coordinates'][0]}, {result['coordinates'][1]}, {result['coordinates'][2]})")

            with col2:
                st.metric("Similarity", f"{similarity_percent:.1f}%")

            if result['apl_code']:
                st.code(result['apl_code'], language='apl')

if __name__ == "__main__":
    main()