"""
toy_data.py -- a named, multi-domain catalog written in the pipeline's format.

EmbeddingGemma is gated, so for a fully offline (and reproducible) demo we
synthesize several clearly different domains with readable product names and
content-clustered embeddings, then write them where prepare_data.py +
embed_items.py would:

    data/items.jsonl     {"item", "asin", "text", "category"}   (text is tagged)
    data/sequences.json  {"train": [[item ids...]], "user_ids": [...]}
    data/item_emb.npy     (n_items, embed_dim) float32, L2-normalized

The real rqvae.py / train.py then run on it unchanged. The embedding structure
is hierarchical -- domain center + a *strong* brand sub-signal + a weak edition
sub-signal + a little noise -- so (a) the top Semantic ID code separates
domains, deeper codes separate brands, and (b) brand-coherent user histories
give the recommender something real to learn.

If you've produced real Gemma embeddings, nothing here overwrites them.
"""

import json

import numpy as np

# domain -> (product types, brands). Items are unique "Brand Product [Edition]".
CATALOG = {
    "Musical_Instruments": (
        ["Dreadnought Guitar", "Nylon Guitar", "Digital Piano", "Synthesizer",
         "Snare Drum", "Hi-Hat Cymbals", "Condenser Microphone", "Overdrive Pedal",
         "Tube Amplifier", "Instrument Cable"],
        ["Yamaha", "Fender", "Roland", "Shure", "Boss", "Pearl", "Korg", "Zildjian",
         "Gibson", "Casio"],
    ),
    "Video_Games": (
        ["Wireless Controller", "Gaming Headset", "RPG Disc", "Racing Wheel",
         "Charging Dock", "Battery Pack", "Capture Card", "Fight Stick",
         "VR Headset", "Memory Card"],
        ["Sony", "Microsoft", "Nintendo", "Logitech", "Razer", "PowerA", "8BitDo",
         "HyperX", "SteelSeries", "Turtle Beach"],
    ),
    "Office_Products": (
        ["Gel Pen Pack", "Stapler", "Sticky Notes", "Laser Printer", "Ink Cartridge",
         "Desk Organizer", "File Folders", "Whiteboard Markers", "Paper Shredder",
         "Label Maker"],
        ["BIC", "Sharpie", "HP", "Brother", "Post-it", "Avery", "Swingline", "Pilot",
         "Epson", "Fellowes"],
    ),
    "Home_Kitchen": (
        ["Chef Knife", "Nonstick Pan", "Blender", "Coffee Maker", "Toaster",
         "Cutting Board", "Mixing Bowls", "Food Storage Set", "Stand Mixer",
         "Espresso Machine"],
        ["Cuisinart", "KitchenAid", "OXO", "Ninja", "Pyrex", "Lodge", "Breville",
         "Hamilton Beach", "Tefal", "Instant Pot"],
    ),
    "Sports_Outdoors": (
        ["Yoga Mat", "Dumbbell Set", "Running Shoes", "Camping Tent", "Water Bottle",
         "Bike Helmet", "Resistance Bands", "Sleeping Bag", "Trekking Poles",
         "Foam Roller"],
        ["Nike", "Adidas", "TheNorthFace", "Coleman", "Hydro Flask", "Under Armour",
         "Garmin", "Columbia", "REI", "Decathlon"],
    ),
    "Pet_Supplies": (
        ["Dog Leash", "Cat Litter Box", "Pet Bed", "Chew Toy", "Aquarium Filter",
         "Bird Cage", "Dog Food Bag", "Scratching Post", "Pet Carrier",
         "Grooming Brush"],
        ["PetSafe", "Kong", "Purina", "ChuckIt", "Fluval", "Frisco", "Hills",
         "Outward Hound", "Catit", "Nylabone"],
    ),
}

EDITIONS = ["", "Pro", "Lite", "Mk II", "2024", "XL", "Mini"]


def domains():
    return list(CATALOG)


def build_catalog(n_per_category=600, embed_dim=128, seed=0):
    """Return (names, categories, brands, embeddings).
    `brands` ("domain|brand") lets us build brand-coherent user histories."""
    rng = np.random.default_rng(seed)
    centers = {c: rng.normal(size=embed_dim) for c in CATALOG}
    names, categories, brand_lbl, vecs = [], [], [], []
    for cat, (bases, brands) in CATALOG.items():
        brand_off = {b: 0.55 * rng.normal(size=embed_dim) for b in brands}  # strong
        edit_off = {e: 0.10 * rng.normal(size=embed_dim) for e in EDITIONS}  # weak
        combos = [(b, base, e) for b in brands for base in bases for e in EDITIONS]
        rng.shuffle(combos)
        for brand, base, ed in combos[:n_per_category]:
            names.append(f"{brand} {base}{(' ' + ed) if ed else ''}")
            categories.append(cat)
            brand_lbl.append(f"{cat}|{brand}")
            vecs.append(centers[cat] + brand_off[brand] + edit_off[ed]
                        + 0.12 * rng.normal(size=embed_dim))
    emb = np.stack(vecs).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return names, categories, brand_lbl, emb


def make_sequences(groups, n_users=12000, seq_len=12, seed=0):
    """Brand-coherent histories: each user sticks to one brand-within-domain, so
    the next item is predictable from content. List order stands in for time ->
    last item = test target, second-to-last = validation."""
    rng = np.random.default_rng(seed + 1)
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    usable = [g for g, items in by_group.items() if len(items) >= 3]
    seqs = []
    for _ in range(n_users):
        g = rng.choice(usable)
        seqs.append([int(i) for i in rng.choice(by_group[g], size=seq_len)])
    return seqs


def _tagged(name, cat):
    domain = cat.replace("_", " ").lower()
    return (f"<item_name> {name} </item_name> "
            f"<category> {cat} </category> "
            f"<description> {name}, a {domain} product. </description>")


def write_artifacts(data_dir, names, categories, emb, sequences):
    data_dir = str(data_dir)
    with open(f"{data_dir}/items.jsonl", "w") as f:
        for i, (name, cat) in enumerate(zip(names, categories)):
            f.write(json.dumps({"item": i, "asin": f"TOY{i:05d}",
                                "text": _tagged(name, cat), "category": cat}) + "\n")
    with open(f"{data_dir}/sequences.json", "w") as f:
        json.dump({"train": sequences, "user_ids": [f"u{i}" for i in range(len(sequences))]}, f)
    np.save(f"{data_dir}/item_emb.npy", emb)
