"""Two-row paper figure, generated exclusively from paired OOF exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def choose_case(frame):
    acf = frame.loc[(frame.model == "acf") & frame.active & frame.image_complete].copy()
    successful = acf.loc[acf.predicted_grade == acf.reference_grade].copy()
    success = len(successful) > 0
    pool = successful if success else acf
    if pool.empty:
        raise ValueError("No active complete held-out case exists for the planned figure")
    pool = pool.assign(distance=(pool.brier-pool.brier.median()).abs())
    return pool.sort_values(["distance","subject_id"]).iloc[0].subject_id, success


def render_case(out, frame, manifest):
    sid, success = choose_case(frame)
    records = {r["subject_id"]: r for r in manifest["rows"]}
    record = records[sid]["record"]
    with np.load(out/"cache"/f"{sid}.npz") as cached:
        images, masks = cached["images"], cached["masks"]
        provenance = json.loads(str(cached["provenance"]))
    fig, axes = plt.subplots(2,7,figsize=(17,7.5),gridspec_kw={"width_ratios":[1,1,1,1,2.15,1.65,2.2]})
    fig.subplots_adjust(left=.035,right=.99,top=.85,bottom=.20,wspace=.23,hspace=.62)
    fig.suptitle(f"{sid} | Same patient, evidence, and output rule", fontsize=20, y=.98, fontweight="bold")
    fig.text(.5,.915, "Successful held-out example" if success else "Held-out failure analysis: no eligible ACF success",
             ha="center",fontsize=14)
    selected = []
    for row_index, variant in enumerate(("acf","learned_joint")):
        patient = frame.loc[(frame.subject_id==sid)&(frame.model==variant)]
        if len(patient)!=1:
            raise ValueError("Figure requires exactly one paired held-out prediction per model")
        row = patient.iloc[0]
        states_path = out/f"fold{int(row.fold)}"/variant/"states.npz"
        with np.load(states_path) as state_file:
            index = np.flatnonzero(state_file["subject_id"] == sid)
            if len(index)!=1:
                raise ValueError("Missing designated state for figure")
            mask = state_file["mask"][index[0],0]
            state = state_file["designated_state"][index[0]]
            if bool(state_file["active"][index[0]]) != bool(row.active):
                raise AssertionError("State audit and prediction disagree")
        fig.text(.035,.87 if row_index==0 else .46,"ACF" if row_index==0 else "Learned-joint + same projection",
                 fontsize=17,fontweight="bold",color="#126f87" if row_index==0 else "#805435")
        for col, name in enumerate(("T1","T1ce","T2","FLAIR")):
            ax = axes[row_index,col]
            ax.imshow(images[col,3].T,cmap="gray",origin="lower",vmin=0,vmax=1)
            if col==3:
                mask2d=masks[:,3].sum(0).T>0
                if mask2d.any() and (~mask2d).any():
                    ax.contour(mask2d,levels=[.5],colors=["#22b8b0"],linewidths=.7)
            ax.set_title(name,fontsize=15)
            ax.axis("off")
        card=axes[row_index,4]
        card.axis("off")
        method=record["IDH_Testing Method"]
        card.text(0,1, "Molecular report",fontweight="bold",fontsize=15,va="top")
        card.text(0,.85, f'IDH: {record["IDH"]} ({method})\n1p/19q: {record["1p19Q CODEL"]}\n'
                  f'MGMT: {record["MGMT"]}\nScope eligible: yes\n\nAuthority: report\nActive: p(Grade 4) = 0\nAllowed: grades 2, 3',
                  fontsize=14,va="top",linespacing=1.2)
        ax=axes[row_index,5]
        # Actual dependency mask, including support rows; not learned attention scores.
        ax.imshow(mask,cmap="Blues",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title("Dependency mask",fontsize=14)
        ax.set_xticks(range(9),["T1","CE","T2","FL","M","S1","S2","S3","S4"],rotation=90,fontsize=10)
        ax.set_yticks([0,1,2,3,4,5,6,7,8],["T1","CE","T2","FL","M","S1","S2","S3","S4"],fontsize=10)
        ax.set_xlabel("Sender",fontsize=12)
        ax.set_ylabel("Receiver",fontsize=12)
        ax.axhline(3.5,color="#e6a126",lw=1)
        ax.axhline(4.5,color="#e6a126",lw=1)
        ax=axes[row_index,6]
        x=np.arange(3)
        p0=np.array([row[f"raw_p{g}"] for g in (2,3,4)])
        p=np.array([row[f"p{g}"] for g in (2,3,4)])
        ax.bar(x-.17,p0,.34,color="#abb6c4",label="Before projection")
        ax.bar(x+.17,p,.34,color="#167e9a",label="After projection")
        for i,value in enumerate(p):
            ax.text(i+.17,value+.035,f"{value:.3f}",ha="center",fontsize=13,fontweight="bold")
        ax.set_xticks(x,["Grade 2","Grade 3","Grade 4"],fontsize=12)
        ax.set_ylim(0,1.15)
        ax.set_yticks([0,.5,1],["0","0.5","1"],fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(f'Pred: {int(row.predicted_grade)} | Ref: {int(row.reference_grade)}',fontsize=15,fontweight="bold")
        ax.text(0,-.20,f'Brier: {row.brier:.4f} | Δh (N/O)\n'
                f'{row.noise_shift:.2e} / {row.occlusion_shift:.2e}',transform=ax.transAxes,fontsize=12,va="top")
        if row_index==0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=11, loc="upper right", bbox_to_anchor=(.99,.94),
                       frameon=False, ncol=2)
        selected.append(dict(model=variant, prediction=row.to_dict(), actual_mask=mask.tolist(),
                             designated_state=state.tolist(), states_file=str(states_path.relative_to(out))))
    fig.text(.035,.015,"M: molecular report; S1–S4: read-only supports. Segmentation contour localizes the tumor. "
             "Δh: designated-state shift; N/O: controlled MRI noise/occlusion.",fontsize=11)
    fig.savefig(out/"utsw_authority_case.pdf",bbox_inches="tight")
    fig.savefig(out/"utsw_authority_case.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    selection=dict(subject_id=sid, success_example=success,
                   selection="Active, complete, ACF-correct OOF cases; nearest median ACF Brier, tie by ID. "
                             "If none correct, same rule on failures.",
                   provenance=provenance, molecular_record=record, results=selected)
    (out/"figure_provenance.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    return selection


def export_results(output):
    out=Path(output)
    frame=pd.read_csv(out/"oof_predictions.csv")
    manifest=json.loads((out/"manifest.json").read_text())
    selection=render_case(out,frame,manifest)
    stats=json.loads((out/"summary.json").read_text())
    a,b=stats["acf"],stats["learned_joint"]
    body=("\\paragraph{Molecularly constrained UTSW grading.}\n"
          "We evaluate three-class grading (grades 2, 3, and 4) using five patient folds and one training seed. "
          "A WHO 2021-based rule permits grades 2 and 3 for accepted IDH-mutant, 1p/19q-codeleted records "
          "within a prespecified oligodendroglioma scope. The same eligibility bit, evidence, and output projection "
          "are supplied to both models. "
          f'ACF and learned-joint obtain Brier scores of {a["brier"]:.4f} and {b["brier"]:.4f}, '
          f'with accuracies of {100*a["accuracy"]:.1f}\\% and {100*b["accuracy"]:.1f}\\%, respectively. '
          f'The relation is active for {a["active_n"]}/{a["n"]} held-out records; '
          f'AVR is {a["avr"]:.4f} and {b["avr"]:.4f}. '
          "Figure~\\ref{fig:utsw-authority-case} separates grading predictions, rule execution, and designated-state behavior. "
          "This retrospective evaluation uses curated molecular records and the recorded reference grades.\n")
    caption=(f'Paired held-out UTSW example ({selection["subject_id"]}). '
             +("The ACF-correct case is selected by median Brier within eligible successes; "
                if selection["success_example"] else "No eligible ACF success exists; a failure case is shown. ")
             +"the baseline receives identical evidence and the same output constraint. The mask shows permitted dependencies. "
             "MRI noise and occlusion assess state protection under fixed molecular evidence. "
             "The reference grade is read separately from the dataset. Full-cohort results determine performance.")
    tex="{\\color{blue}\n"+body+"}\n\\begin{figure}[t]\n\\centering\n" \
        +"\\includegraphics[width=\\linewidth]{utsw_authority_case.pdf}\n" \
        +"\\caption{\\textcolor{blue}{"+caption+"}}\n\\label{fig:utsw-authority-case}\n\\end{figure}\n"
    (out/"paper_insert_blue.tex").write_text(tex,encoding="utf-8")
    (out/"paper_results.md").write_text(body.replace("\\%","%").replace("\\paragraph", "")+"\n"+caption,encoding="utf-8")
    appendix=Path(__file__).resolve().parents[1]/"config"/"utsw_authority_appendix.tex"
    (out/"paper_appendix_blue.tex").write_text(appendix.read_text(encoding="utf-8"),encoding="utf-8")
    names={"acf":"ACF", "learned_joint":"Learned-joint + projection", "prior":"Label prior",
           "prior_projected":"Label prior + projection", "molecular_projected":"Molecular + projection"}
    order=["prior","prior_projected","molecular_projected","learned_joint","acf"]
    headers=["Model","Brier","Accuracy","AVR","Residual Brier","Residual Accuracy"]
    keys=["brier","accuracy","avr","residual_brier","residual_accuracy"]
    table_rows=[]
    for name in order:
        table_rows.append([names[name]]+["N/A" if stats[name].get(key) is None else f"{stats[name][key]:.4f}" for key in keys])
    best={key:(max if "accuracy" in key else min)(stats[name][key] for name in order if stats[name].get(key) is not None)
          for key in keys if any(stats[name].get(key) is not None for name in order)}
    def highlighted(row, tex=False):
        cells=[row[0]]
        for key, value in zip(keys,row[1:]):
            mark=key in best and value==f"{best[key]:.4f}"
            cells.append(("\\textbf{"+value+"}" if tex else "**"+value+"**") if mark else value)
        return cells
    caption_table="Full-cohort out-of-fold grading results. Brier is dimension-normalized; AVR uses activated relations."
    markdown="**Table 1: "+caption_table+"**\n\n|"+"|".join(headers)+"|\n|"+"|".join(["---"]*len(headers))+"|\n"
    markdown+="\n".join("|"+"|".join(highlighted(row))+"|" for row in table_rows)
    markdown+="\n\nState-shift and denominator details: summary.json. Paired patient intervals: paired_bootstrap.json.\n"
    (out/"results_table.md").write_text(markdown,encoding="utf-8")
    table_tex="\\begin{table}[t]\n\\centering\n{\\color{blue}\n\\caption{"+caption_table+"}\n"
    table_tex+="\\small\n\\begin{tabular}{lrrrrr}\n\\hline\n"+" & ".join(["Model","Brier","Acc.","AVR","Res. Brier","Res. Acc."])+" \\\\\n\\hline\n"
    table_tex+="\n".join(" & ".join(highlighted(row,tex=True))+" \\\\" for row in table_rows)
    table_tex+="\n\\hline\n\\end{tabular}\n}\n\\end{table}\n"
    (out/"results_table_blue.tex").write_text(table_tex,encoding="utf-8")
