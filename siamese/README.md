# Simese Model Codebase


<table>
    <tr>
      <th>Training data creation </th>
      <td>
<pre>
$ sh script_training_data.sh  
# (internally contains) [python3 create_training_data.py "/path/pdfdirectorylocation" /path/siamese/trainingdata 200 250]
</pre>

To Generate summary file
<pre>
$ python3 generate_combination_file.py /efsdata/siamese/trainingdata 1000

$ shuf -n 4000 inventory_eval.txt > inventory_eval.txt_4000
</pre>

 </td>
  </tr>
  
  <tr>
  <th>Training </th>
  <td>
<pre>
$ python application.py --mode TRAIN --datafile inventory_train.txt,inventory_eval.txt_4000
</pre>

  </td>
  </tr>
  
  <tr>
  <th>Evaluation</th>
  <td>
<pre>
$ python application.py --mode EVAL --datafile inventory_eval.txt_2000 --modeldir "models_step=34200_loss=0.06"
</pre>

  </td>
  </tr>
  
  <tr>
  <th>Prediction</th>
  <td>
<pre>
$ python application.py --mode PREDICT --modeldir "models_step=34200_loss=0.06" --predfiles "/nfsdata/type1/pdf1.pdf,/nfsdata/type2/pdf1.pdf"

$ python application.py --mode PREDICT --modeldir "models_step=34200_loss=0.06" --datafile predictfile.txt --outputfile result.csv
</pre>
  </td>
  </tr>
</table>


