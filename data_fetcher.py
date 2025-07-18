import cellxgene_census
import scanpy as sc
import numpy as np
import argparse

def quote_values(values):
    """
        Format a list of string values into a filter-compatible string for use with SOMA queries.
        Example: ['lung', 'kidney'] -> "['lung', 'kidney']"
    """
    
    return "[" + ", ".join(f"'{v}'" for v in values) + "]"

def main():
    parser = argparse.ArgumentParser(description="Fetch cellxgene data based on user parameters")

    # Data Options (allowing multiple values)
    parser.add_argument("--species", type=str, default="Homo sapiens", help="Species name (currently only Homo sapiens is supported)")
    parser.add_argument("--cell_type", nargs="+", default=["lung"], help="Tissue types (e.g., lung kidney)")
    parser.add_argument("--sex", nargs="+", default=None, help="Donor sexes: male female")
    parser.add_argument("--disease", nargs="+", default=['normal','lung adenocarcinoma', 'squamous cell lung carcinoma', 'small cell lung carcinoma', 'non-small cell lung carcinoma', 'pleomorphic carcinoma', 'lung large cell carcinoma'], help="Diseases: COVID-19 cancer etc.")
    parser.add_argument("--n_samples", type=int, default=60000, help="Number of cells to receive, if available")

    # Save Options
    parser.add_argument("--out_file", type=str, default="data/data.h5ad", help="Path to save the output AnnData file")

    args = parser.parse_args()

    species_key = args.species.lower().replace(" ", "_")

    print("Opening cellxgene census...")
    with cellxgene_census.open_soma() as census:
        filters = ["is_primary_data == True"]

        # Cell type
        if args.cell_type:
            filters.append(f"tissue_general in {quote_values(args.cell_type)}")

        # Sex
        if args.sex:
            filters.append(f"sex in {quote_values(args.sex)}")

        # Disease
        if args.disease:
            filters.append(f"disease in {quote_values(args.disease)}")

        full_filter = " and ".join(filters)
        print(f"Using filter: {full_filter}")
        print("Fetching observation metadata...")

        obs_df = census["census_data"][species_key].obs.read(
            value_filter=full_filter
        ).concat().to_pandas()

        print(f"Total matching cells: {len(obs_df)}")
        if obs_df.empty:
            print("No matching cells found with the given filters.")
            return

        # Sample (or not)
        sample_size = min(args.n_samples, len(obs_df))
        df_sample = obs_df.sample(n=sample_size, random_state=42)
        ids_str = "[" + ",".join(str(i) for i in df_sample["soma_joinid"].tolist()) + "]"

        print(f"Downloading {sample_size} cells...")

        adata = cellxgene_census.get_anndata(
            census=census,
            organism=args.species,
            obs_value_filter=f"soma_joinid in {ids_str}",
            obs_column_names=["tissue", "disease", "sex", "cell_type", "tissue_general"],
        )

        print(f"Download complete. Writing to {args.out_file}...")
        adata.write(args.out_file)
        print("Done.")

if __name__ == "__main__":
    main()