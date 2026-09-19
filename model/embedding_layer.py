# from model.synapse1_11_mini import SynapseBrain

# model = SynapseBrain.from_pretrained(f"synapse_brain")

def analyze_embedding_ratio(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    embed = 0

    # token embedding
    embed += (
        model.get_input_embeddings()
        .weight.numel()
    )

    # input projection
    embed += (
        model.embedding
        .embed_proj.weight.numel()
    )

    # output projection
    embed += (
        model.lm_head
        .weight.numel()
    )

    print(f"Total: {total:,}")
    print(f"FeedForward-related: {total-embed:,}")
    print(f"Embedding-related: {embed:,}")
    print(f"Embedding-Ratio: {embed/total:.2%}")

# analyze_embedding_ratio(model)