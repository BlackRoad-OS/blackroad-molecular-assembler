#!/usr/bin/env python3
"""
BlackRoad Molecular Assembler
Molecular assembly simulation system for drug discovery and materials science.
"""

import json
import math
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Atom:
    """Represents an atom in 3D space."""
    element: str
    x: float
    y: float
    z: float
    charge: float


@dataclass
class Bond:
    """Represents a bond between two atoms."""
    atom1: int
    atom2: int
    type: str  # "single"|"double"|"triple"|"aromatic"
    length_angstrom: float


@dataclass
class Molecule:
    """Represents a complete molecule."""
    id: str
    name: str
    formula: str
    atoms: List[Dict]
    bonds: List[Dict]
    mass_g_mol: float
    charge: float
    energy_kj_mol: float
    created_at: str


class MolecularAssembler:
    """Main molecular assembly and simulation controller."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the molecular assembler."""
        if db_path is None:
            db_path = os.path.expanduser("~/.blackroad/molecules.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.molecules: Dict[str, Molecule] = {}
        self._init_db()
        self._load_library_presets()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS molecules
                     (id TEXT PRIMARY KEY, name TEXT, formula TEXT, 
                      atoms TEXT, bonds TEXT, mass_g_mol REAL, 
                      charge REAL, energy_kj_mol REAL, created_at TEXT)''')
        conn.commit()
        conn.close()

    def _load_library_presets(self):
        """Load common molecules into library."""
        presets = {
            "H2O": {
                "name": "Water",
                "formula": "H2O",
                "atoms": [
                    {"element": "O", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.34},
                    {"element": "H", "x": 0.96, "y": 0.0, "z": 0.0, "charge": 0.17},
                    {"element": "H", "x": -0.24, "y": 0.93, "z": 0.0, "charge": 0.17},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "single", "length_angstrom": 0.96},
                    {"atom1": 0, "atom2": 2, "type": "single", "length_angstrom": 0.96},
                ],
                "mass": 18.015,
                "charge": 0.0,
                "energy": -10.2,
            },
            "CO2": {
                "name": "Carbon Dioxide",
                "formula": "CO2",
                "atoms": [
                    {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0, "charge": 0.8},
                    {"element": "O", "x": 1.16, "y": 0.0, "z": 0.0, "charge": -0.4},
                    {"element": "O", "x": -1.16, "y": 0.0, "z": 0.0, "charge": -0.4},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "double", "length_angstrom": 1.16},
                    {"atom1": 0, "atom2": 2, "type": "double", "length_angstrom": 1.16},
                ],
                "mass": 44.01,
                "charge": 0.0,
                "energy": -23.1,
            },
            "CH4": {
                "name": "Methane",
                "formula": "CH4",
                "atoms": [
                    {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.36},
                    {"element": "H", "x": 0.63, "y": 0.63, "z": 0.63, "charge": 0.09},
                    {"element": "H", "x": -0.63, "y": -0.63, "z": 0.63, "charge": 0.09},
                    {"element": "H", "x": -0.63, "y": 0.63, "z": -0.63, "charge": 0.09},
                    {"element": "H", "x": 0.63, "y": -0.63, "z": -0.63, "charge": 0.09},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "single", "length_angstrom": 1.09},
                    {"atom1": 0, "atom2": 2, "type": "single", "length_angstrom": 1.09},
                    {"atom1": 0, "atom2": 3, "type": "single", "length_angstrom": 1.09},
                    {"atom1": 0, "atom2": 4, "type": "single", "length_angstrom": 1.09},
                ],
                "mass": 16.04,
                "charge": 0.0,
                "energy": -17.9,
            },
            "C6H6": {
                "name": "Benzene",
                "formula": "C6H6",
                "atoms": [
                    {"element": "C", "x": 1.4, "y": 0.0, "z": 0.0, "charge": -0.05},
                    {"element": "C", "x": 0.7, "y": 1.21, "z": 0.0, "charge": -0.05},
                    {"element": "C", "x": -0.7, "y": 1.21, "z": 0.0, "charge": -0.05},
                    {"element": "C", "x": -1.4, "y": 0.0, "z": 0.0, "charge": -0.05},
                    {"element": "C", "x": -0.7, "y": -1.21, "z": 0.0, "charge": -0.05},
                    {"element": "C", "x": 0.7, "y": -1.21, "z": 0.0, "charge": -0.05},
                    {"element": "H", "x": 2.48, "y": 0.0, "z": 0.0, "charge": 0.05},
                    {"element": "H", "x": 1.24, "y": 2.15, "z": 0.0, "charge": 0.05},
                    {"element": "H", "x": -1.24, "y": 2.15, "z": 0.0, "charge": 0.05},
                    {"element": "H", "x": -2.48, "y": 0.0, "z": 0.0, "charge": 0.05},
                    {"element": "H", "x": -1.24, "y": -2.15, "z": 0.0, "charge": 0.05},
                    {"element": "H", "x": 1.24, "y": -2.15, "z": 0.0, "charge": 0.05},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "aromatic", "length_angstrom": 1.40},
                    {"atom1": 1, "atom2": 2, "type": "aromatic", "length_angstrom": 1.40},
                    {"atom1": 2, "atom2": 3, "type": "aromatic", "length_angstrom": 1.40},
                    {"atom1": 3, "atom2": 4, "type": "aromatic", "length_angstrom": 1.40},
                    {"atom1": 4, "atom2": 5, "type": "aromatic", "length_angstrom": 1.40},
                    {"atom1": 5, "atom2": 0, "type": "aromatic", "length_angstrom": 1.40},
                ],
                "mass": 78.11,
                "charge": 0.0,
                "energy": -29.5,
            },
            "NH3": {
                "name": "Ammonia",
                "formula": "NH3",
                "atoms": [
                    {"element": "N", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.36},
                    {"element": "H", "x": 0.94, "y": 0.0, "z": 0.0, "charge": 0.12},
                    {"element": "H", "x": -0.47, "y": 0.81, "z": 0.0, "charge": 0.12},
                    {"element": "H", "x": -0.47, "y": -0.41, "z": 0.70, "charge": 0.12},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "single", "length_angstrom": 1.01},
                    {"atom1": 0, "atom2": 2, "type": "single", "length_angstrom": 1.01},
                    {"atom1": 0, "atom2": 3, "type": "single", "length_angstrom": 1.01},
                ],
                "mass": 17.03,
                "charge": 0.0,
                "energy": -11.3,
            },
            "O2": {
                "name": "Oxygen",
                "formula": "O2",
                "atoms": [
                    {"element": "O", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.4},
                    {"element": "O", "x": 1.21, "y": 0.0, "z": 0.0, "charge": -0.4},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "double", "length_angstrom": 1.21},
                ],
                "mass": 32.00,
                "charge": 0.0,
                "energy": -5.1,
            },
            "N2": {
                "name": "Nitrogen",
                "formula": "N2",
                "atoms": [
                    {"element": "N", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.4},
                    {"element": "N", "x": 1.10, "y": 0.0, "z": 0.0, "charge": -0.4},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "triple", "length_angstrom": 1.10},
                ],
                "mass": 28.01,
                "charge": 0.0,
                "energy": -9.8,
            },
            "C2H5OH": {
                "name": "Ethanol",
                "formula": "C2H5OH",
                "atoms": [
                    {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0, "charge": -0.18},
                    {"element": "C", "x": 1.54, "y": 0.0, "z": 0.0, "charge": -0.03},
                    {"element": "O", "x": 2.38, "y": 1.20, "z": 0.0, "charge": -0.35},
                    {"element": "H", "x": 2.38, "y": -0.87, "z": 0.0, "charge": 0.06},
                    {"element": "H", "x": -0.37, "y": 0.94, "z": 0.0, "charge": 0.06},
                    {"element": "H", "x": -0.37, "y": -0.47, "z": 0.82, "charge": 0.06},
                    {"element": "H", "x": -0.37, "y": -0.47, "z": -0.82, "charge": 0.06},
                    {"element": "H", "x": 3.31, "y": 1.20, "z": 0.0, "charge": 0.19},
                ],
                "bonds": [
                    {"atom1": 0, "atom2": 1, "type": "single", "length_angstrom": 1.54},
                    {"atom1": 1, "atom2": 2, "type": "single", "length_angstrom": 1.43},
                    {"atom1": 2, "atom2": 7, "type": "single", "length_angstrom": 0.96},
                ],
                "mass": 46.07,
                "charge": 0.0,
                "energy": -18.5,
            },
        }

        for mol_id, data in presets.items():
            self.add_molecule(mol_id, data["name"], data["formula"],
                            data["atoms"], data["bonds"],
                            data["mass"], data["charge"], data["energy"])

    def add_molecule(self, mol_id: str, name: str, formula: str,
                    atoms: List[Dict], bonds: List[Dict],
                    mass: float = 0.0, charge: float = 0.0,
                    energy: float = 0.0):
        """Register a molecule."""
        now = datetime.now().isoformat()
        mol = Molecule(
            id=mol_id,
            name=name,
            formula=formula,
            atoms=atoms,
            bonds=bonds,
            mass_g_mol=mass,
            charge=charge,
            energy_kj_mol=energy,
            created_at=now,
        )
        self.molecules[mol_id] = mol
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            '''INSERT OR REPLACE INTO molecules 
               (id, name, formula, atoms, bonds, mass_g_mol, charge, energy_kj_mol, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (mol_id, name, formula, json.dumps(atoms), json.dumps(bonds),
             mass, charge, energy, now)
        )
        conn.commit()
        conn.close()

    def assemble(self, mol_ids: List[str], reaction_type: str) -> str:
        """Combine molecules to create a product."""
        if len(mol_ids) < 2:
            raise ValueError("Need at least 2 molecules to assemble")
        
        mols = [self.molecules[mid] for mid in mol_ids]
        
        # Simple combination: concatenate atoms and bonds
        combined_atoms = []
        combined_bonds = []
        atom_offset = 0
        
        combined_mass = 0.0
        combined_charge = 0.0
        
        for mol in mols:
            combined_atoms.extend(mol.atoms)
            combined_mass += mol.mass_g_mol
            combined_charge += mol.charge
            
            for bond in mol.bonds:
                combined_bonds.append({
                    "atom1": bond["atom1"] + atom_offset,
                    "atom2": bond["atom2"] + atom_offset,
                    "type": bond["type"],
                    "length_angstrom": bond["length_angstrom"],
                })
            atom_offset += len(mol.atoms)
        
        product_id = f"product_{reaction_type}_{len(self.molecules)}"
        product_name = f"{reaction_type.capitalize()} Product"
        product_formula = "+".join([m.formula for m in mols])
        
        combined_energy = sum(m.energy_kj_mol for m in mols) - 5.0  # Reaction energy
        
        self.add_molecule(product_id, product_name, product_formula,
                         combined_atoms, combined_bonds,
                         combined_mass, combined_charge, combined_energy)
        
        return product_id

    def simulate_energy(self, mol_id: str) -> float:
        """Calculate Lennard-Jones potential (simplified)."""
        mol = self.molecules[mol_id]
        energy = 0.0
        
        for i, atom1 in enumerate(mol.atoms):
            for j, atom2 in enumerate(mol.atoms[i+1:], start=i+1):
                dx = atom2["x"] - atom1["x"]
                dy = atom2["y"] - atom1["y"]
                dz = atom2["z"] - atom1["z"]
                r = math.sqrt(dx**2 + dy**2 + dz**2) + 0.1  # Avoid division by zero
                
                # Lennard-Jones: 4*eps*((sigma/r)^12 - (sigma/r)^6)
                sigma = 3.4  # Angstroms
                eps = 0.1  # kcal/mol
                energy += 4 * eps * ((sigma/r)**12 - (sigma/r)**6)
        
        return energy

    def optimize_geometry(self, mol_id: str, iterations: int = 10) -> float:
        """Iterative position refinement via mock gradient descent."""
        mol = self.molecules[mol_id]
        learning_rate = 0.01
        
        for _ in range(iterations):
            for i, atom in enumerate(mol.atoms):
                # Calculate forces from neighboring atoms
                fx, fy, fz = 0.0, 0.0, 0.0
                
                for j, other in enumerate(mol.atoms):
                    if i == j:
                        continue
                    dx = other["x"] - atom["x"]
                    dy = other["y"] - atom["y"]
                    dz = other["z"] - atom["z"]
                    r = math.sqrt(dx**2 + dy**2 + dz**2) + 0.1
                    
                    # Repulsive force
                    force_mag = 0.1 / (r**2)
                    fx += force_mag * (dx / r)
                    fy += force_mag * (dy / r)
                    fz += force_mag * (dz / r)
                
                # Update positions
                atom["x"] += learning_rate * fx
                atom["y"] += learning_rate * fy
                atom["z"] += learning_rate * fz
        
        final_energy = self.simulate_energy(mol_id)
        mol.energy_kj_mol = final_energy
        return final_energy

    def get_properties(self, mol_id: str) -> Dict:
        """Get molecular properties."""
        mol = self.molecules[mol_id]
        return {
            "id": mol.id,
            "name": mol.name,
            "formula": mol.formula,
            "mass_g_mol": mol.mass_g_mol,
            "charge": mol.charge,
            "energy_kj_mol": mol.energy_kj_mol,
            "num_atoms": len(mol.atoms),
            "num_bonds": len(mol.bonds),
            "created_at": mol.created_at,
        }

    def search(self, query: str) -> List[Dict]:
        """Search molecules by formula or name."""
        results = []
        query_lower = query.lower()
        
        for mol in self.molecules.values():
            if (query_lower in mol.formula.lower() or
                query_lower in mol.name.lower()):
                results.append(self.get_properties(mol.id))
        
        return results

    def export_xyz(self, mol_id: str) -> str:
        """Export molecule in XYZ format."""
        mol = self.molecules[mol_id]
        lines = [str(len(mol.atoms)), f"Generated from {mol.name}\n"]
        
        for atom in mol.atoms:
            lines.append(
                f"{atom['element']:>2}  {atom['x']:10.6f}  {atom['y']:10.6f}  {atom['z']:10.6f}"
            )
        
        return "\n".join(lines)

    def export_json(self, mol_id: str) -> str:
        """Export molecule as JSON."""
        mol = self.molecules[mol_id]
        return json.dumps(asdict(mol), indent=2)

    def library_presets(self) -> List[str]:
        """Get list of library preset molecule IDs."""
        return ["H2O", "CO2", "CH4", "C6H6", "NH3", "O2", "N2", "C2H5OH"]


def main():
    """CLI interface."""
    import sys
    
    assembler = MolecularAssembler()
    
    if len(sys.argv) < 2:
        print("Usage: molecular_assembler.py [library|assemble|export]")
        return
    
    command = sys.argv[1]
    
    if command == "library":
        presets = assembler.library_presets()
        print(f"Available molecules ({len(presets)}):")
        for mol_id in presets:
            props = assembler.get_properties(mol_id)
            print(f"  {mol_id:12} {props['name']:20} {props['formula']}")
    
    elif command == "assemble" and len(sys.argv) >= 4:
        mol1 = sys.argv[2]
        mol2 = sys.argv[3]
        reaction = sys.argv[4] if len(sys.argv) > 4 else "synthesis"
        product = assembler.assemble([mol1, mol2], reaction)
        props = assembler.get_properties(product)
        print(f"Product created: {product}")
        print(f"Formula: {props['formula']}, Mass: {props['mass_g_mol']:.2f} g/mol")
    
    elif command == "export" and len(sys.argv) >= 3:
        mol_id = sys.argv[2]
        fmt = sys.argv[3] if len(sys.argv) > 3 else "xyz"
        if fmt == "xyz":
            print(assembler.export_xyz(mol_id))
        elif fmt == "json":
            print(assembler.export_json(mol_id))
    
    elif command == "optimize" and len(sys.argv) >= 3:
        mol_id = sys.argv[2]
        energy = assembler.optimize_geometry(mol_id)
        print(f"Optimized {mol_id}, final energy: {energy:.2f} kJ/mol")
    
    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        results = assembler.search(query)
        print(f"Search results for '{query}':")
        for mol in results:
            print(f"  {mol['id']:12} {mol['name']:20} {mol['formula']}")
    
    else:
        print("Unknown command or invalid arguments")


if __name__ == "__main__":
    main()
