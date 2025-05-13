class Shifter:
    def shift(self, susbstr_merged):
        begin_index = { # begin_index_of_descriptor
            "count_atoms_in_molecule": 0,
            "count_atoms": 12,
            "count_bond_types": 13,
            "count_all_bonds": 17,
            "calculate_wiener_index": 18,
            "categorize_logp_detailed": 19,
            "categorize_uff_energy": 20,
            "count_ring_atoms": 21,
            "count_complete_rings": 22
        }
        m = {
            "count_atoms_in_molecule": 100,
            "count_atoms": 50,
            "count_bond_types": 300,
            "count_all_bonds": 300,
            "calculate_wiener_index": 5000,
            "categorize_logp_detailed": 8,
            "categorize_uff_energy": 9,
            "count_ring_atoms": 50,
            "count_complete_rings": 20
        }
        for substructure in susbstr_merged:
            if substructure == '$':
                continue
            for j in range(len(substructure)):
                if substructure[6][0] == 'Invalid': # КОСТЫЛЬ
                    substructure[6][0] = 8
                    
                if j == 0:
                    for i in range(begin_index["count_atoms_in_molecule"], begin_index["count_atoms"]):
                        substructure[j][i] += i * m["count_atoms_in_molecule"]
                elif j == 1:
                    substructure[j][0] += 12 * m["count_atoms_in_molecule"]
                elif j == 2:
                    for i in range(begin_index["count_bond_types"], begin_index["count_all_bonds"]):
                        substructure[j][i - begin_index["count_bond_types"]] += (i - begin_index["count_bond_types"]) * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 3:
                    substructure[j][0] += 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 4:
                    substructure[j][0] += m["count_all_bonds"] + 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 5:
                    substructure[j][0] += m["calculate_wiener_index"] + m["count_all_bonds"] + 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 6:
                    substructure[j][0] += m["categorize_logp_detailed"] + m["calculate_wiener_index"] + m["count_all_bonds"] + 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 7:
                    substructure[j][0] += m["categorize_uff_energy"] + m["categorize_logp_detailed"] + m["calculate_wiener_index"] + m["count_all_bonds"] + 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
                elif j == 8:
                    substructure[j][0] += m["count_ring_atoms"] + m["categorize_uff_energy"] + m["categorize_logp_detailed"] + m["calculate_wiener_index"] + m["count_all_bonds"] + 4 * m["count_bond_types"] + m["count_atoms"] + 12 * m["count_atoms_in_molecule"]
            #break
        
        for substructure in susbstr_merged: 
            if substructure == '$':
                continue
            for j in range(len(substructure)):
                for i in range(len(substructure[j])):
                    substructure[j][i] += 6 # FOR SPECIAL TOKENS
                    
        return susbstr_merged