from rdkit.Chem import BRICS
from rdkit import Chem
import numpy as np

class Splitter:
    def __init__(self):
        self.i = 0
    
    def _get_broken_bonds(self, mol: Chem.rdchem.Mol):
        brics_bonds = list(BRICS.FindBRICSBonds(mol))

        broken_bonds = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
        return broken_bonds
    
    def _mol_to_adjacency_matrix(self, mol: Chem.rdchem.Mol):
        """Конвертирует молекулу RDKit в матрицу смежности"""
        num_atoms = mol.GetNumAtoms()

        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_type = bond.GetBondTypeAsDouble() # (одинарная=1, двойная=2 и т.д.)

            adj_matrix[i][j] = bond_type
            adj_matrix[j][i] = bond_type
        return adj_matrix
    
    def _get_masked_adjacency_matrix(self, mol: Chem.rdchem.Mol, broken_bonds: list[tuple[int]]):
        matrix = self._mol_to_adjacency_matrix(mol)
        for pair in broken_bonds:
            a,b = pair
            matrix[a][b] = -1
            matrix[b][a] = -1
        return matrix
    
    def _find_connected_components(self, adj_matrix: np.array):
        """
        Находит все компоненты связности в графе, заданном матрицей смежности.
        Возвращает список компонент, где каждая компонента - это список вершин.
        """
        n = len(adj_matrix) 
        visited = [False] * n 

        def dfs(v, component):
            visited[v] = True
            component.append(v)
            for u in range(n):
                if adj_matrix[v][u] != 0 and adj_matrix[v][u] != -1 and not visited[u]:
                    dfs(u, component)

        components = []
        for v in range(n):
            if not visited[v]:
                component = []
                dfs(v, component)
                components.append(component)

        return components
    
    def _get_merged_substructures(self, broken_bonds, list_of_atoms_substructures): 
        # для каждого ребра, по которому исходно дробили молекулу, склеивает 2 соседние подструктуры в одну
        merged_substr = []
        cnt = 0
        for pair in broken_bonds:
            a,b = pair
            res_substr = []
            for substr in list_of_atoms_substructures:
                if a in substr or b in substr:
                    res_substr += substr
                    cnt += 1
                if cnt == 2: # попытка оптимизировать, мы склеиваем только 2 подструктуры, если уже их нашли -- выходим из цикла
                    cnt = 0
                    break
            cnt = 0
            merged_substr.append(res_substr)
        return merged_substr
    
    def _get_smiles_of_substructures(self, mol, list_of_connected_components):
        sub_smiles = []
        for atom_indices in list_of_connected_components:
            emol = Chem.EditableMol(Chem.Mol())

            # Map original atom indices to new atom indices
            atom_map = {}
            for i, idx in enumerate(atom_indices):
                atom = mol.GetAtomWithIdx(idx)
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_idx = emol.AddAtom(new_atom)
                atom_map[idx] = new_idx

            # Add only the bonds between atoms in the substructure
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_map and a2 in atom_map:
                    emol.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

            submol = emol.GetMol()
            submol = Chem.RemoveHs(submol)
            Chem.SanitizeMol(submol)
            smiles = Chem.MolToSmiles(submol)
            sub_smiles.append(smiles)

        return sub_smiles

    def get_substructures_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)
        
        broken_bonds = self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol, broken_bonds)
        
        list_of_connected_components = self._find_connected_components(adj_matrix)
        list_of_smiles = self._get_smiles_of_substructures(mol, list_of_connected_components)
        
        return list_of_smiles
        
    def get_substructures_merged(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)
        
        broken_bonds = self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol, broken_bonds)
        
        list_of_connected_components = self._find_connected_components(adj_matrix)
        merged_substructures = self._get_merged_substructures(broken_bonds, list_of_connected_components)
        
        return merged_substructures
        
    def get_substructures_smiles_and_merged(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)
        
        broken_bonds = self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol, broken_bonds)
        
        list_of_connected_components = self._find_connected_components(adj_matrix)
        list_of_smiles = self._get_smiles_of_substructures(mol, list_of_connected_components)
        merged_substructures = self._get_merged_substructures(broken_bonds, list_of_connected_components)
        
        list_of_smiles.append('$') # разделяющий символ между химически-осознанными подструктурами и склееными
        list_of_smiles += self._get_smiles_of_substructures(mol, merged_substructures)
        return list_of_smiles