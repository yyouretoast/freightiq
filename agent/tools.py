from langchain_core.tools import tool
from rag.retriever import retrieve_carriers_semantic, query_carriers_sql

@tool
def carrier_semantic_search(query: str) -> str:
    """
    Search the carrier database semantically using natural language queries. 
    Best for loose matches, regional capabilities, equipment matching, or qualitative descriptions 
    (e.g., 'flatbed hauling in the midwest', 'refrigerated carriers in Ohio').
    """
    results = retrieve_carriers_semantic(query, k=5)
    return "\n\n---\n\n".join(results) if results else "No carrier profiles matched your semantic query."

@tool
def carrier_sql_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query on the 'carriers' table to retrieve structured information. 
    Highly recommended for exact searches, filtering by safety rating, specific states, dot_number, 
    mc_number, or counting metrics.
    
    Columns in 'carriers' table:
    - id (INTEGER)
    - carrier_name (TEXT)
    - dot_number (TEXT)
    - mc_number (TEXT)
    - hq_state (TEXT)
    - service_regions (TEXT - comma-separated list, e.g., 'Midwest, Southwest')
    - equipment_types (TEXT - comma-separated list, e.g., 'dry van, reefer')
    - cargo_specializations (TEXT - comma-separated list, e.g., 'hazardous materials')
    - safety_rating (TEXT - 'satisfactory', 'conditional', or 'unsatisfactory')
    - years_operating (INTEGER)
    - contact_email (TEXT)
    - notes (TEXT)
    
    Example queries:
    - SELECT * FROM carriers WHERE dot_number = '1234567'
    - SELECT carrier_name FROM carriers WHERE hq_state = 'TX' AND safety_rating = 'satisfactory'
    """
    return query_carriers_sql(query)

@tool
def web_search(query: str) -> str:
    """
    Query the web for current freight rates, market trends, external carrier news, 
    and real-time logistics or shipping industry data.
    """
    from duckduckgo_search import DDGS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web search results found."
        return "\n\n".join([f"Source: {r.get('title')}\nLink: {r.get('href')}\nContent: {r.get('body')}" for r in results])
    except Exception as e:
        return f"Error executing web search: {str(e)}"

@tool
def freight_class_calculator(weight_lbs: float, length_in: float, width_in: float, height_in: float) -> str:
    """
    Calculate the NMFC freight class based on shipment weight in pounds and dimensions in inches.
    Accurately maps density (lbs/cubic foot) to the standard NMFC classification (50 - 500).
    """
    if weight_lbs <= 0 or length_in <= 0 or width_in <= 0 or height_in <= 0:
        return "Error: All inputs (weight, length, width, height) must be greater than zero."
        
    cubic_inches = length_in * width_in * height_in
    cubic_feet = cubic_inches / 1728.0
    density = weight_lbs / cubic_feet
    
    if density >= 50:
        freight_class = 50
    elif density >= 35:
        freight_class = 55
    elif density >= 30:
        freight_class = 60
    elif density >= 22.5:
        freight_class = 65
    elif density >= 15:
        freight_class = 70
    elif density >= 13.5:
        freight_class = 77.5
    elif density >= 12:
        freight_class = 85
    elif density >= 10.5:
        freight_class = 92.5
    elif density >= 9:
        freight_class = 100
    elif density >= 8:
        freight_class = 110
    elif density >= 7:
        freight_class = 125
    elif density >= 6:
        freight_class = 150
    elif density >= 5:
        freight_class = 175
    elif density >= 4:
        freight_class = 200
    elif density >= 3:
        freight_class = 250
    elif density >= 2:
        freight_class = 300
    elif density >= 1:
        freight_class = 400
    else:
        freight_class = 500
        
    return (
        f"Shipment Dimensions: {length_in}x{width_in}x{height_in} inches\n"
        f"Volume: {cubic_feet:.2f} cubic feet\n"
        f"Weight: {weight_lbs} lbs\n"
        f"Calculated Density: {density:.2f} lb/ft³\n"
        f"Standard NMFC Freight Class: {freight_class}"
    )

tools = [carrier_semantic_search, carrier_sql_query, web_search, freight_class_calculator]
