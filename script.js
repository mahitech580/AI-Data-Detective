const features = [
  "Dataset profiling",
  "Data quality detection",
  "Missing-value analysis",
  "Outlier detection",
  "Statistical analysis",
  "Correlation analysis",
  "Machine learning",
  "Visualization generation",
  "Automated reports"
];

const root = document.getElementById("features");

features.forEach((feature,index)=>{
  const item=document.createElement("div");
  item.className="feature";
  item.textContent=
    String(index+1).padStart(2,"0")+"  "+feature;
  root.appendChild(item);
});
