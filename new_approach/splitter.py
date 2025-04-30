from rdkit.Chem import BRICS
from rdkit import Chem
import numpy as np

class Splitter:
    def __init__(self):
        self.i = 0
        self.matrix = []
        self.broken_bonds = []
        self.count = 0

    def _get_broken_bonds(self, mol: Chem.rdchem.Mol):
        brics_bonds = list(BRICS.FindBRICSBonds(mol))

        broken_bonds = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
        self.broken_bonds = broken_bonds

    def _mol_to_adjacency_matrix(self, mol: Chem.rdchem.Mol):
        """Конвертирует молекулу RDKit в матрицу смежности"""
        num_atoms = mol.GetNumAtoms()

        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_type = int(bond.GetBondTypeAsDouble()) # (одинарная=1, двойная=2 и т.д.)

            adj_matrix[i][j] = bond_type
            adj_matrix[j][i] = bond_type
        return adj_matrix

    def _get_subgraph_adjacency_matrix(self, full_adj_matrix, subgraph_vertices):
        """ Возвращает матрицу смежности подграфа

        Параметры:
        full_adj_matrix - исходная матрица смежности (numpy array или список списков)
        subgraph_vertices - список вершин подграфа (индексы вершин в исходном графе)

        Возвращает:
        Матрицу смежности подграфа (numpy array)
        """
        # Преобразуем в numpy array для удобства работы
        full_adj = np.array(full_adj_matrix)

        # Получаем подматрицу только для вершин подграфа
        sub_adj = full_adj[np.ix_(subgraph_vertices, subgraph_vertices)]

        return sub_adj

    def _get_masked_adjacency_matrix(self, mol: Chem.rdchem.Mol):
        matrix = self._mol_to_adjacency_matrix(mol)
        for a,b in self.broken_bonds:
            matrix[a][b] = -1
            matrix[b][a] = -1
        #print(self.broken_bonds, "!")
        self.matrix = matrix
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

    def _get_merged_substructures(self, list_of_atoms_substructures):
        # для каждого ребра, по которому исходно дробили молекулу, склеивает 2 соседние подструктуры в одну
        merged_substr = []
        cnt = 0
        for pair in self.broken_bonds:
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

        from rdkit.Chem import rdmolops

    def _get_substructure_smiles_fragments_not_merged(self, mol, components): # работает только для непересекающихся подструктур (кажется)
        # Создаем редактируемую копию
        emol = Chem.RWMol(mol)

        # Удаляем все связи между компонентами
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            in_same = any(a1 in comp and a2 in comp for comp in components)
            if not in_same:
                emol.RemoveBond(a1, a2)

                for i in range(int(mol.GetBondBetweenAtoms(a1, a2).GetBondTypeAsDouble())):
                    new_atom1 = emol.AddAtom(Chem.Atom(1))
                    emol.AddBond(a1, new_atom1, Chem.rdchem.BondType.SINGLE)

                    new_atom2 = emol.AddAtom(Chem.Atom(1))
                    emol.AddBond(a2, new_atom2, Chem.rdchem.BondType.SINGLE)

        # Разделяем на фрагменты
        frags = Chem.GetMolFrags(emol.GetMol(), asMols=True)
        return [Chem.MolToSmiles(Chem.RemoveHs(frag)) for frag in frags]

    '''def _get_smiles_of_substructures(self, mol, list_of_connected_components):
        try:
            sub_smiles = []
            #print(self.broken_bonds)
            for atom_indices in list_of_connected_components:
                # Create an editable molecule from the original
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
                #print(self.matrix)
                # adding hydrogens instead of broken bonds
                #print(self.broken_bonds)
                #print(mol.GetAtomWithIdx(11).GetAtomicNum())
                #print(atom_indices)
                for v, u in self.broken_bonds:
                    #print(u,v, mol.GetBondBetweenAtoms(v, u).GetBondTypeAsDouble())
                    #print(self.matrix[v][u])
                    for i in range(int(mol.GetBondBetweenAtoms(v, u).GetBondTypeAsDouble())):
                        for atom_idx in atom_indices:
                            if atom_idx == v:
                                new_atom = emol.AddAtom(Chem.Atom(1))
                                emol.AddBond(atom_map[v], new_atom, Chem.rdchem.BondType.SINGLE)
                            elif atom_idx == u:
                                new_atom = emol.AddAtom(Chem.Atom(1))
                                emol.AddBond(atom_map[u], new_atom, Chem.rdchem.BondType.SINGLE)

                submol = emol.GetMol()
                #Chem.SanitizeMol(submol)
                #print(self._mol_to_adjacency_matrix(submol))
                #submol = Chem.AddHs(submol)
                submol = Chem.RemoveHs(submol)
                Chem.SanitizeMol(submol)
                smiles = Chem.MolToSmiles(submol)
                sub_smiles.append(smiles)

            return sub_smiles
        except:
            self.count += 1
            return []
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

        return sub_smiles'''

    '''def get_substructures_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)

        broken_bonds = self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol, broken_bonds)

        list_of_connected_components = self._find_connected_components(adj_matrix)
        list_of_smiles = self._get_smiles_of_substructures(mol, list_of_connected_components)

        return list_of_smiles'''

    from rdkit.Chem import rdmolops

    def _removing_bonds(self, current_component, adj_matr):
        removing_bonds = []
        for i in range(len(adj_matr)):
            for j in range(i, len(adj_matr)):

                if adj_matr[i][j] != 0:
                    if i in current_component and j not in current_component:
                        removing_bonds.append((i,j))

                    elif i not in current_component and j in current_component:
                        removing_bonds.append((i,j))
        '''for a1, a2 in bonds:
            for comp in components:
                if comp == current_component or (set(comp) & set(current_component)):
                    #print(a1, a2)
                    continue
                if (a1 in current_component and a2 in comp) or (a1 in comp and a2 in current_component):
                    print(a1, a2, comp, current_component)
                    removing_bonds.append((a1, a2))'''
        return removing_bonds

    def _get_true_substr(self, adj_matr, frags, component):
        true_adj_matr = self._get_subgraph_adjacency_matrix(adj_matr, component)
        #print(component, '!', true_adj_matr)
        for frag in frags:
            #print(mol_to_adjacency_matrix(frag))
            if np.array_equal(self._mol_to_adjacency_matrix(Chem.RemoveHs(frag)), true_adj_matr):
                #print('gpp')
                #print(Chem.MolToSmiles(frag))
                return frag

    def _get_substructure_smiles_fragments_merged(self, mol, components): # работает только для непересекающихся подструктур (кажется)
        # Создаем редактируемую копию
        adj_matr = self._mol_to_adjacency_matrix(mol)
        bonds = []
        smiles_of_frags = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            bonds.append((a1,a2))

        # Удаляем все связи между компонентами
        for component in components:
            removing_bnds = []
            removing_bnds = self._removing_bonds(component, adj_matr)
            emol = Chem.RWMol(mol)
            #print(removing_bnds)
            for a1, a2 in removing_bnds:
                emol.RemoveBond(a1, a2)

                for i in range(int(mol.GetBondBetweenAtoms(a1, a2).GetBondTypeAsDouble())):
                    #break # debug, should be removed
                    new_atom1 = emol.AddAtom(Chem.Atom(1))
                    emol.AddBond(a1, new_atom1, Chem.rdchem.BondType.SINGLE)

                    new_atom2 = emol.AddAtom(Chem.Atom(1))
                    emol.AddBond(a2, new_atom2, Chem.rdchem.BondType.SINGLE)

            # print(find_connected_components(mol_to_adjacency_matrix(emol.GetMol())), component, components)
            frags = Chem.GetMolFrags(emol.GetMol(), asMols=True)
            #print([Chem.MolToSmiles(Chem.RemoveHs(frag)) for frag in frags])
            true_frag = self._get_true_substr(adj_matr, frags, component)
            smi = Chem.MolToSmiles(Chem.RemoveHs(true_frag))
            smiles_of_frags.append(smi)
            #return 0
        return smiles_of_frags

    def _get_substructures_merged(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)

        broken_bonds = self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol, broken_bonds)

        list_of_connected_components = self._find_connected_components(adj_matrix)
        merged_substructures = self._get_merged_substructures(broken_bonds, list_of_connected_components)

        return merged_substructures

    def get_substructures_smiles_and_merged(self, smiles: str):
        #print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        mol.UpdatePropertyCache(strict=True)

        self._get_broken_bonds(mol)
        adj_matrix = self._get_masked_adjacency_matrix(mol)


        list_of_connected_components = self._find_connected_components(adj_matrix)
        #print(list_of_connected_components)
        list_of_connected_components = [sorted(i) for i in list_of_connected_components]

        list_of_smiles = self._get_substructure_smiles_fragments_not_merged(mol, list_of_connected_components)
        merged_substructures = self._get_merged_substructures(list_of_connected_components)

        merged_substructures = [sorted(i) for i in merged_substructures]

        list_of_smiles.append('$') # разделяющий символ между химически-осознанными подструктурами и склееными
        list_of_smiles += self._get_substructure_smiles_fragments_merged(mol, merged_substructures)
        return list_of_smiles