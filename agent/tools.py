import logging
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from rag.retriever import retrieve_carriers_semantic, query_carriers_sql
import config

logger = logging.getLogger(__name__)

@tool
def carrier_semantic_search(query: str) -> str:
    """
    Search the carrier database using natural language and vector similarity.
    Returns the top-k most semantically relevant carrier profiles for the given query.
    """
    results = retrieve_carriers_semantic(query, k=config.SEMANTIC_RETRIEVAL_K)
    return "\n\n---\n\n".join(results) if results else "No carrier profiles matched your semantic query."

@tool
def carrier_sql_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query on the 'carriers' table.
    
    Columns:
    - id (INTEGER)
    - carrier_name (TEXT)
    - dot_number (TEXT)
    - mc_number (TEXT)
    - hq_state (TEXT) — use exact match: hq_state = 'OH'
    - service_regions (TEXT — JSON array, e.g. '["Midwest", "Southwest"]')
    - equipment_types (TEXT — JSON array, e.g. '["dry van", "flatbed"]')
    - cargo_specializations (TEXT — JSON array, e.g. '["hazardous materials"]')
    - safety_rating (TEXT — 'satisfactory', 'conditional', or 'unsatisfactory')
    - years_operating (INTEGER)
    - contact_email (TEXT)
    - notes (TEXT)
    
    For JSON array columns use json_each() for exact matching, or LIKE for partial:
    - Exact:   EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest')
    - Partial: service_regions LIKE '%Midwest%'
    
    Examples:
    - SELECT * FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory'
    - SELECT * FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'hazardous materials')
    - SELECT * FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')
    """
    # Security: Only allow SELECT and WITH queries, reject anything else
    clean = query.strip().rstrip(";").strip()
    upper_clean = clean.upper()
    if not (upper_clean.startswith("SELECT") or upper_clean.startswith("WITH")):
        return "Error: Only SELECT queries are permitted on the carriers database."
    
    # Enforce an upper bound of 25 rows via subquery wrapping
    # Guarantees limit enforcement regardless of inner clauses, aliases, or string content
    wrapped_query = f"SELECT * FROM ({clean}) AS _bounded_carriers LIMIT 25"
    
    return query_carriers_sql(wrapped_query)

@tool
def web_search(query: str) -> str:
    """
    Query the web for current freight rates, market trends, external carrier news, 
    and real-time logistics or shipping industry data.
    """
    try:
        results = []
        with DDGS() as ddgs:
            try:
                results = list(ddgs.text(query, max_results=3))
            except Exception as e:
                logger.debug(f"DDGS text search failed, falling back to news: {e}")
            if not results:
                results = list(ddgs.news(query, max_results=3))
        if not results:
            return "No web search results found for this query."
        formatted = []
        for r in results:
            title = r.get("title", "")
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or r.get("snippet") or ""
            source = r.get("source") or ""
            prefix = f"Source: {source}\n" if source else ""
            formatted.append(f"{prefix}Title: {title}\nLink: {url}\nContent: {body}".strip())
        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Web search is temporarily unavailable due to upstream network limits. Do not retry web search; answer the query directly based on available information or state that live market search is currently unavailable."

@tool
def freight_class_calculator(weight_lbs: float, length_in: float, width_in: float, height_in: float, cargo_description: str = "") -> str:
    """
    Calculate the NMFC freight class based on shipment weight in pounds, dimensions in inches, and optional cargo description.
    Accurately maps density (lbs/cubic foot) to standard NMFC class, or resolves fixed class exceptions (e.g. insulation).
    """
    if weight_lbs <= 0 or length_in <= 0 or width_in <= 0 or height_in <= 0:
        return "Error: All inputs (weight, length, width, height) must be greater than zero."
        
    cubic_inches = length_in * width_in * height_in
    cubic_feet = cubic_inches / 1728.0
    density = weight_lbs / cubic_feet
    
    # LTL Exceptions Check
    exceptions = {
        "insulation": 150,
        "bulk mail": 70,
        "raw mail": 70,
        "ping pong balls": 500,
        "plastic cups": 250
    }
    
    applied_exception = None
    if cargo_description:
        desc_lower = cargo_description.lower()
        for keyword, ex_class in exceptions.items():
            if keyword in desc_lower:
                applied_exception = (keyword, ex_class)
                break
    
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
        
    if applied_exception:
        keyword, ex_class = applied_exception
        return (
            f"Shipment Dimensions: {length_in}x{width_in}x{height_in} inches\n"
            f"Volume: {cubic_feet:.2f} cubic feet\n"
            f"Weight: {weight_lbs} lbs\n"
            f"Calculated Density: {density:.2f} lb/ft³\n"
            f"Density-Based NMFC Class (pre-exception): {freight_class}\n"
            f"LTL EXCEPTION RULE APPLIED: Cargo contains '{keyword}' — fixed NMFC Class {ex_class} overrides density calculation."
        )

    return (
        f"Shipment Dimensions: {length_in}x{width_in}x{height_in} inches\n"
        f"Volume: {cubic_feet:.2f} cubic feet\n"
        f"Weight: {weight_lbs} lbs\n"
        f"Calculated Density: {density:.2f} lb/ft³\n"
        f"Standard NMFC Freight Class: {freight_class}"
    )

tools = [carrier_semantic_search, carrier_sql_query, web_search, freight_class_calculator]
