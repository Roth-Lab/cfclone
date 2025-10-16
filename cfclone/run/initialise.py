import juliapkg


def initialise():
    juliapkg.require_julia("1.11.5", ".")
    juliapkg.resolve()

    juliapkg.add("ADTypes")
    juliapkg.add("Bijectors")
    juliapkg.add("BridgeStan")
    juliapkg.add("Distributions")
    juliapkg.add("DynamicPPL")
    juliapkg.add("JSON")
    juliapkg.add("Pigeons")
    juliapkg.add("ReverseDiff")
    juliapkg.resolve()
