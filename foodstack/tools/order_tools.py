"""Order management tools for the FoodStack agents."""

from langchain_core.tools import tool

from foodstack.data.orders import order_db

@tool
def get_order_status(search_value: str ) -> str:
    """Get the status of an order by order ID, tracking ID, or email.

    At least one search parameter must be provided.

    Args:
        order_id: The order ID (e.g., 'ORD-001').
        tracking_id: The tracking ID (e.g., 'FS202TRK').
        email: The customer email address.

    Returns:
        A formatted string with order details and status, or an error message.
    """
    if not search_value:
        return "Error: Please provide an order ID, tracking ID, or email address."

    matching_orders = []
    search_value = search_value.upper()

    # Search by order ID
    if search_value in order_db:
        order = order_db[search_value]
        order["order_id"] = search_value
        matching_orders.append(order)

    # Search by tracking ID or email
    for order_id, order in order_db.items():
        if order["tracking_id"].upper() == search_value or order["customer_email"].lower() == search_value.lower():
            order["order_id"] = order_id
            if order not in matching_orders:
                matching_orders.append(order)

    if not matching_orders:
        return "No orders found matching the provided criteria."

    # Format results
    results = []
    for order in matching_orders:
        formatted_order = f"""
            Order ID: {order['order_id']}
            Tracking ID: {order['tracking_id']}
            Customer Name: {order['customer_name']}
            Customer Email: {order['customer_email']}
            Status: {order['status']}
            Price: {order['price']}
            Ordered On: {order['order_date']}
            Estimated delivery: {order['estimated_delivery']}
        """

    return "\n" + "=" * 50 + "\n".join(results) + "\n" + "=" * 50


# Export tools for agent use
order_tools = [get_order_status]
