#!/usr/bin/env python3
"""Task 2 — generate (question, gold_sql) pairs against sql.db.

Templates cover count / filter / aggregate / join / order+limit / distinct.
Parameters are drawn from the DB's actual contents so every gold query runs and
(for filters) returns something. Each gold SQL is executed at generation time;
if it errors it is dropped. Train/val use one paraphrase pool; --hard uses a
disjoint pool of phrasings the model never trained on.

Rows: {"q": question, "sql": gold_sql}. eval scores by EXECUTION — run both,
compare the rows — so surface differences in the SQL don't matter, only results.
"""
import sqlite3, json, random, argparse

def sampler(db):
    con = sqlite3.connect(db); cur = con.cursor()
    cities = [r[0] for r in cur.execute("SELECT DISTINCT city FROM customers")]
    countries = [r[0] for r in cur.execute("SELECT DISTINCT country FROM customers")]
    cats = [r[0] for r in cur.execute("SELECT DISTINCT category FROM products")]
    stats = [r[0] for r in cur.execute("SELECT DISTINCT status FROM orders")]
    names = [r[0] for r in cur.execute(
        "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id=c.id")]
    con.close()
    return dict(city=cities, country=countries, cat=cats, status=stats, name=names,
               price=[50,100,200,300,500], amt=[100,300,500,1000,2000], n=[3,5])

# each template: sql(params) and two phrasing pools
TEMPLATES = [
  dict(
    sql=lambda p: "SELECT COUNT(*) FROM orders",
    q=["how many orders are there in total?", "count all orders", "what is the total number of orders?"],
    qh=["give me the count of every order on file", "how many orders exist altogether?"],
    p=lambda s: {}),
  dict(
    sql=lambda p: f"SELECT COUNT(*) FROM orders WHERE status = '{p['s']}'",
    q=["how many {s} orders are there?", "count the {s} orders", "number of orders with status {s}"],
    qh=["how many orders came back as {s}?", "tally the orders marked {s}"],
    p=lambda s: {"s": random.choice(s["status"])}),
  dict(
    sql=lambda p: f"SELECT name FROM customers WHERE city = '{p['c']}'",
    q=["list the names of customers in {c}", "which customers are based in {c}? names only",
       "names of customers from {c}"],
    qh=["who are our customers located in {c}? just names", "give the customer names for {c}"],
    p=lambda s: {"c": random.choice(s["city"])}),
  dict(
    sql=lambda p: f"SELECT COUNT(*) FROM customers WHERE country = '{p['c']}'",
    q=["how many customers are from {c}?", "count customers in country {c}"],
    qh=["what's the headcount of customers in {c}?", "number of {c} customers"],
    p=lambda s: {"c": random.choice(s["country"])}),
  dict(
    sql=lambda p: f"SELECT name FROM products WHERE category = '{p['cat']}' AND price < {p['pr']}",
    q=["names of {cat} products under {pr}", "which {cat} products cost less than {pr}? names",
       "list {cat} products cheaper than {pr}"],
    qh=["show me {cat} items priced below {pr}, names only", "{cat} products with price under {pr}?"],
    p=lambda s: {"cat": random.choice(s["cat"]), "pr": random.choice(s["price"])}),
  dict(
    sql=lambda p: "SELECT SUM(total) FROM orders",
    q=["what is the total revenue across all orders?", "sum of all order totals", "total revenue overall"],
    qh=["add up the value of every order", "what do all orders come to in total?"],
    p=lambda s: {}),
  dict(
    sql=lambda p: f"SELECT SUM(total) FROM orders WHERE status = '{p['s']}'",
    q=["what is the total revenue from {s} orders?", "sum the totals of {s} orders",
       "revenue from orders that are {s}"],
    qh=["how much money is in {s} orders?", "total value of the {s} orders"],
    p=lambda s: {"s": random.choice(s["status"])}),
  dict(
    sql=lambda p: f"SELECT AVG(price) FROM products WHERE category = '{p['cat']}'",
    q=["what is the average price of {cat} products?", "average {cat} price", "mean price in {cat}"],
    qh=["on average, what does a {cat} product cost?", "typical price of {cat} items?"],
    p=lambda s: {"cat": random.choice(s["cat"])}),
  dict(
    sql=lambda p: f"SELECT name FROM products ORDER BY price DESC LIMIT {p['n']}",
    q=["name the {n} most expensive products", "top {n} products by price, names",
       "which {n} products are priced highest? names"],
    qh=["give me the {n} priciest products by name", "the {n} dearest items, names only"],
    p=lambda s: {"n": random.choice(s["n"])}),
  dict(
    sql=lambda p: f"SELECT COUNT(*) FROM orders WHERE total > {p['a']}",
    q=["how many orders have a total above {a}?", "count orders over {a}", "orders with total greater than {a}?"],
    qh=["how many orders exceeded {a} in value?", "number of orders worth more than {a}"],
    p=lambda s: {"a": random.choice(s["amt"])}),
  dict(
    sql=lambda p: ("SELECT COUNT(*) FROM orders JOIN customers "
                   f"ON orders.customer_id = customers.id WHERE customers.name = '{p['nm']}'"),
    q=["how many orders did {nm} place?", "count the orders by {nm}", "number of orders from {nm}"],
    qh=["how many purchases has {nm} made?", "order count for the customer {nm}"],
    p=lambda s: {"nm": random.choice(s["name"])}),
  dict(
    sql=lambda p: "SELECT DISTINCT city FROM customers",
    q=["list all distinct cities customers are from", "what cities do customers live in?",
       "unique customer cities"],
    qh=["which cities appear among our customers?", "give the set of customer cities"],
    p=lambda s: {}),
]

def fmt(params):  # map template param keys to phrasing placeholders
    return {"s": params.get("s"), "c": params.get("c"), "cat": params.get("cat"),
            "pr": params.get("pr"), "n": params.get("n"), "a": params.get("a"),
            "nm": params.get("nm")}

def make_row(s, con, hard):
    t = random.choice(TEMPLATES)
    params = t["p"](s)
    sql = t["sql"](params)
    try:
        con.execute(sql).fetchall()
    except Exception:
        return None
    pool = t["qh"] if hard else t["q"]
    q = random.choice(pool).format(**{k: v for k, v in fmt(params).items() if v is not None})
    return {"q": q, "sql": sql}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="train/2_sql/sql.db")
    ap.add_argument("--out", default="train/2_sql/data")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--hard", type=int, default=200)
    args = ap.parse_args()
    import os; os.makedirs(args.out, exist_ok=True)
    s = sampler(args.db)
    con = sqlite3.connect(args.db)

    # the template space is finite, so 'unique' stops after a bounded number of
    # attempts and returns what it found; train allows repeats (fine for SFT).
    def batch(n, hard, unique):
        seen, rows, tries = set(), [], 0
        while len(rows) < n and tries < n * 400:
            tries += 1
            r = make_row(s, con, hard)
            if not r: continue
            if unique:
                if r["q"] in seen: continue
                seen.add(r["q"])
            rows.append(r)
        return rows

    for name, n, hard, uniq in [("train", args.train, False, False),
                                ("val", args.val, False, True),
                                ("hard", args.hard, True, True)]:
        rows = batch(n, hard, uniq)
        with open(f"{args.out}/{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows")
        for r in rows[:3]:
            print("   ", r["q"], "=>", r["sql"])

if __name__ == "__main__":
    main()
