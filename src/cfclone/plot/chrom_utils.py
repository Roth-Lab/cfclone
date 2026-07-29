# sort_chroms adapted from: https://github.com/Roth-Lab/hapclone-smk/blob/main/scripts/plot_clone_pseudobulk.py
def sort_chroms(chroms):
    numeric = []
    string = []

    if chroms[0].startswith("chr"):
        chr_prefix = True
    else:
        chr_prefix = False

    for c in chroms:
        if chr_prefix:
            c = c.replace("chr", "")
        try:
            numeric.append(int(c))
        except ValueError:
            string.append(c)

    chroms = [str(x) for x in sorted(numeric)] + list(sorted(string))

    if chr_prefix:
        chroms = ["chr{}".format(x) for x in chroms]

    return chroms
