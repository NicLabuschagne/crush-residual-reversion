import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd


    return (pd,)


@app.cell
def _(pd):
    df = pd.read_parquet(r"C:\Users\Amin\PycharmProjects\PythonProject1\data\raw\soybn_usd_1m.parquet")
    df.isna().sum()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
