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