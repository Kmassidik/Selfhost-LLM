#!/usr/bin/env python3
"""Task 2 — build a small, deterministic SQLite shop database to run SQL against.

Execution accuracy needs a real DB: score by running gold and predicted SQL and
comparing the rows they return. Seeded, so the DB is identical every rebuild.
Writes sql.db and schema.sql (the CREATE statements the model is shown).
"""
import sqlite3, random, argparse, os

random.seed(42)

CITIES = [("Berlin","DE"),("Munich","DE"),("Paris","FR"),("Lyon","FR"),
          ("Madrid","ES"),("Rome","IT"),("Oslo","NO"),("Vienna","AT"),
          ("Lisbon","PT"),("Dublin","IE")]
CATEGORIES = ["electronics","furniture","stationery","kitchen","garden"]
STATUSES = ["paid","pending","cancelled","refunded"]
FIRST = ["Sam","Priya","Marcus","Lena","Diego","Amara","Tom","Yuki","Nadia","Carl",
         "Ravi","Elise","Omar","Grace","Hans","Mei","Ivan","Sofia","Ben","Zoe",
         "Aiko","Bruno","Carmen","Dev","Esra","Farid","Gita","Hugo","Ines","Jonas"]
LAST = ["Ng","Patel","Reid","Cole","Vega","Osei","Frey","Sato","Khan","Diaz",
        "Roy","Blum","Aziz","Park","Voss","Lim","Petrov","Marin","Lowe","Tan"]
PRODUCTS = {
    "electronics": ["Laptop","Monitor","Keyboard","Mouse","Webcam","Headset","Router","Tablet"],
    "furniture":   ["Desk","Chair","Bookshelf","Sofa","Stool"],
    "stationery":  ["Notebook","Pen Set","Stapler","Binder","Marker Pack"],
    "kitchen":     ["Kettle","Toaster","Blender","Pan Set","Knife Block"],
    "garden":      ["Hose","Trowel","Planter","Shears","Watering Can"],
}

SCHEMA = """CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT,
  city TEXT,
  country TEXT
);
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT,
  category TEXT,
  price REAL
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  product_id INTEGER,
  qty INTEGER,
  total REAL,
  status TEXT,
  order_date TEXT
);"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/2_sql")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = f"{args.out}/sql.db"
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.executescript(SCHEMA)

    # customers
    used = set()
    customers = []
    while len(customers) < 40:
        nm = f"{random.choice(FIRST)} {random.choice(LAST)}"
        if nm in used: continue
        used.add(nm)
        city, country = random.choice(CITIES)
        customers.append((len(customers)+1, nm, city, country))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    # products
    products = []
    for cat, names in PRODUCTS.items():
        for nm in names:
            price = round(random.uniform(9, 999), 2)
            products.append((len(products)+1, nm, cat, price))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    # orders
    orders = []
    for i in range(300):
        c = random.randint(1, len(customers))
        p = random.randint(1, len(products))
        qty = random.randint(1, 5)
        price = products[p-1][3]
        total = round(price * qty, 2)
        status = random.choice(STATUSES)
        month = random.randint(1, 12)
        date = f"2026-{month:02d}-{random.randint(1,28):02d}"
        orders.append((i+1, c, p, qty, total, status, date))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", orders)

    con.commit()
    with open(f"{args.out}/schema.sql", "w") as f:
        f.write(SCHEMA)
    print(f"built {db}: {len(customers)} customers, {len(products)} products, {len(orders)} orders")
    con.close()

if __name__ == "__main__":
    main()
