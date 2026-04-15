def split_into_blocks(matrix, block_size=32):
    height = len(matrix)
    width = len(matrix[0])

    blocks = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):

            block = []

            for bi in range(i, min(i + block_size, height)):
                row = matrix[bi][j:min(j + block_size, width)]
                block.append(row)

            if len(block) == block_size:
                blocks.append(block)

    return blocks


def block_energy(block):
    total = 0
    count = 0

    for row in block:
        for value in row:
            total += value * value
            count += 1

    return total / count


def energy_distribution_variance(blocks):
    energies = [block_energy(b) for b in blocks]

    mean_energy = sum(energies) / len(energies)

    var = sum((e - mean_energy) ** 2 for e in energies) / len(energies)

    return var

def block_variance_stats(blocks):
    variances = []

    for b in blocks:
        m = sum(v for row in b for v in row) / (len(b)*len(b[0]))
        var = sum((v - m)**2 for row in b for v in row) / (len(b)*len(b[0]))
        variances.append(var)

    mean_var = sum(variances) / len(variances)
    var_of_var = sum((v - mean_var)**2 for v in variances) / len(variances)

    return mean_var, var_of_var

def symmetry_score(matrix):
    h = len(matrix)
    w = len(matrix[0])

    diff = 0
    count = 0

    for i in range(h):
        for j in range(w // 2):
            diff += abs(matrix[i][j] - matrix[i][w - j - 1])
            count += 1

    return diff / count

def local_hf_variance(blocks):
    energies = [block_energy(b) for b in blocks]

    mean_e = sum(energies) / len(energies)

    return sum(abs(e - mean_e) for e in energies) / len(energies)

def gradient_direction_consistency(matrix):
    h = len(matrix)
    w = len(matrix[0])

    total = 0
    count = 0

    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = matrix[i][j+1] - matrix[i][j-1]
            gy = matrix[i+1][j] - matrix[i-1][j]

            if gx != 0:
                ratio = abs(gy / gx)
                total += ratio
                count += 1

    return total / (count + 1e-6)

def block_entropy_variance(blocks):
    import math

    entropies = []

    for b in blocks:
        hist = {}
        total = 0

        for row in b:
            for v in row:
                hist[v] = hist.get(v, 0) + 1
                total += 1

        ent = 0
        for c in hist.values():
            p = c / total
            ent -= p * math.log2(p)

        entropies.append(ent)

    mean_e = sum(entropies) / len(entropies)
    var_e = sum((e - mean_e)**2 for e in entropies) / len(entropies)

    return mean_e, var_e