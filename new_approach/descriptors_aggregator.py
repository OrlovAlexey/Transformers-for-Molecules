from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from rdkit import Chem
import networkx as nx

def count_all_bonds(mol): # нужно ли считать вместе с водородами?
    """Возвращает общее количество связей, включая водороды."""
    if mol is None:
        return None
    mol = Chem.AddHs(mol)  # Добавляем водороды
    return mol.GetNumBonds()

COMMON_ATOMS = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'Na', 'K', 'Mg', 'Fe', 'Zn'] # возможно стоит часть выкинуть. Возможно, существенную часть
COMMON_ATOMS_1 = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si', 'Metals'] # возможно стоит часть выкинуть. Возможно, существенную часть

def count_atoms_in_molecule(mol):
    """
    возвращает словарь вида {'C': X, 'H': Y, ..., 'Metals': Z} с количеством атомов каждого типа
              или None, если SMILES битый
    """
    if mol is None:
        return None

    mol = Chem.AddHs(mol)  # Добавляем водороды

    # Инициализируем словарь с нулями для всех значимых атомов
    atom_counts = {atom: 0 for atom in COMMON_ATOMS_1}

    # Считаем атомы каждого типа
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ['Na', 'K', 'Mg', 'Fe', 'Zn']:
            atom_counts['Metals'] += 1
        elif symbol in COMMON_ATOMS_1:  # Остальные атомы
            atom_counts[symbol] += 1

    return atom_counts

def calculate_wiener_index(mol): # подумать над дискретизацией (?)
    """
    Вычисляет индекс Винера
    """
    #mol = Chem.AddHs(mol)
    if mol is None:
        return None

    # Создаём граф молекулы
    G = nx.Graph()

    # Добавляем атомы как вершины графа
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())

    # Добавляем связи как рёбра графа
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        G.add_edge(u, v)

    # Вычисляем индекс Винера (сумма кратчайших путей)
    wiener_index = nx.wiener_index(G)
    return int(wiener_index)


def calculate_logp(mol):
    if mol is None:
        return None
    return rdMolDescriptors._CalcCrippenContribs(mol)[0][0]


def categorize_logp_detailed(logp):
    """7-уровневая дискретизация logP"""
    if logp < -2:
        return 1 # "01_Extreme_hydrophilic"
    elif -2 <= logp < -0.5:
        return 2 # "02_Strong_hydrophilic"
    elif -0.5 <= logp < 0:
        return 3 # "03_Moderate_hydrophilic"
    elif 0 <= logp < 1:
        return 4 # "04_Neutral"
    elif 1 <= logp < 2:
        return 5 # "05_Moderate_hydrophobic"
    elif 2 <= logp < 4:
        return 6 # "06_Strong_hydrophobic"
    else:
        return 7 # "07_Extreme_hydrophobic"
    
def calculate_uff_energy(mol, num_confs=10):
    """
    Вычисляет минимальную энергию UFF для молекулы
    """
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Генерация нескольких конформаций
    energies = []
    for _ in range(num_confs):
        try:
            # Новая конформация
            AllChem.EmbedMolecule(mol, randomSeed=np.random.randint(0,10000))
            # Оптимизация
            ff = AllChem.UFFGetMoleculeForceField(mol)
            ff.Minimize()
            energies.append(ff.CalcEnergy())
        except:
            continue

    return min(energies) if energies else None

def categorize_uff_energy(energy):
    """
    7-уровневая дискретизация энергии UFF с химической интерпретацией.

    Уровни подобраны на основе типичных значений для органических молекул:
    """
    if energy is None:
        return "Invalid"

    if energy < 0:
        return 1 # "01_Stable_complex"
    elif 0 <= energy < 50:
        return 2 # "02_Very_stable"
    elif 50 <= energy < 100:
        return 3 # "03_Stable"
    elif 100 <= energy < 150:
        return 4 # "04_Moderate"
    elif 150 <= energy < 200:
        return 5 # "05_Unstable"
    elif 200 <= energy < 300:
        return 6 # "06_High_energy"
    else:
        return 7 # "07_Extreme_energy"
    
def count_atoms(mol, include_hydrogens=False):
    if mol is None:
        return None

    if include_hydrogens:
        mol = Chem.AddHs(mol)

    return mol.GetNumAtoms()

bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

def count_bond_types(mol):
    """
    Подсчитывает количество связей каждого типа в молекуле.
    """
    if mol is None:
        return None

    bond_counts = {}

    for i in bond_types:
        bond_counts[i] = 0

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.SINGLE:
            bond_counts["SINGLE"] += 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_counts["DOUBLE"] += 1
        elif bond_type == Chem.BondType.TRIPLE:
            bond_counts["TRIPLE"] += 1
        elif bond_type == Chem.BondType.AROMATIC:
            bond_counts["AROMATIC"] += 1

    return bond_counts


def count_ring_atoms(mol):
    """
    Подсчитывает количество атомов, находящихся в циклах (кольцах).
    """
    if mol is None:
        return None

    in_ring = [atom.IsInRing() for atom in mol.GetAtoms()]
    return sum(in_ring)

def count_complete_rings(mol):
    """
    Подсчитывает количество полных колец в молекуле.
    """
    #mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # SSSR (Smallest Set of Smallest Rings)
    sssr = Chem.GetSSSR(mol)
    return len(sssr)


class Aggregator:
    def aggregate(self, smiles): # возвращает полный отпечаток для молекулы
        if smiles is not '$':
            result= []
            mol = Chem.MolFromSmiles(smiles)

            result.append(list(count_atoms_in_molecule(mol).values()))        # 16; max < 50
            result.append([count_atoms(mol)])                                 # max < 50
            result.append(list(count_bond_types(mol).values()))               # 4; max < 300
            result.append([count_all_bonds(mol)])                             # max < 300
            result.append([calculate_wiener_index(mol)])                      # max < 2000
            result.append([categorize_logp_detailed(calculate_logp(mol))])    # max < 8
            result.append([categorize_uff_energy(calculate_uff_energy(mol))]) # max < 8
            result.append([count_ring_atoms(mol)])                            # max < 50
            result.append([count_complete_rings(mol)])                        # max < 10

            return result
        else:
            return smiles