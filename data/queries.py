from sqlalchemy import text
import pandas as pd

def get_churn_raw_data(engine):
    """
    Extracts raw data for churn prediction by joining Customers, Orders, and OrderDetails tables.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        DataFrame with customer_id, order_id, order_date, and total_spent
    """
    query = text("""
        SELECT 
            c.customer_id,
            o.order_id,
            o.order_date,
            SUM(od.unit_price * od.quantity) as total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        GROUP BY c.customer_id, o.order_id, o.order_date
        ORDER BY c.customer_id, o.order_date
    """)
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def get_return_risk_raw_data(engine):
    """
    Extracts raw data for return risk prediction by joining Orders and OrderDetails tables.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        DataFrame with order_id, discount, unit_price, quantity, and total_price
    """
    query = text("""
        SELECT 
            o.order_id,
            od.discount,
            od.unit_price,
            od.quantity,
            (od.unit_price * od.quantity) as total_price
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        ORDER BY o.order_id
    """)
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def get_purchase_raw_data(engine):
    """
    Extracts raw data for purchase prediction by joining Orders, OrderDetails, Products, 
    Categories, and Customers tables.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        DataFrame with customer_id, category_name, and total_spent per category
    """
    query = text("""
        SELECT 
            c.customer_id,
            cat.category_name,
            SUM(od.unit_price * od.quantity) as total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        JOIN categories cat ON p.category_id = cat.category_id
        GROUP BY c.customer_id, cat.category_name
        ORDER BY c.customer_id, cat.category_name
    """)
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn)
