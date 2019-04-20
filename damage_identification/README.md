# Damage Identification
This repo contains code to identify damages on objects like rooftops due to hailsstorms.

## Solutions
<table>
  <tr>
    <td><b>Solution 1</b></td>
	<td>
		<ul>	
			<li>Connected Components</li>
			<li>Classifiers on Simple Features</li>
		</ul>
	</td>
    <td>
		<ul>	
			<li>The approach use Opencv to identify connected regions as potential damages</li>
			<li>During Training Components are manually extracted by asking user Yes/No. </li>
			<li>During training Components marked as damages as saved in csv by extracting features.</li>
			<li>Same features are used for classification during predictions</li>
		</ul>

	</td>
  </tr>
  <tr>
    <td><b>Solution 2</b></td>
	<td>
		<ul>
			<li>Removing Noise</li>
			<li>Sliding Window</li>
			<li>Color spaces</li>
			<li>Classifier on Simple Features</li>
		</ul>
	</td>
    <td>
		<ul>	
			<li>Image is passed to noise filter to remove noise</li>
			<li>Sliding Window of various sizes is used to identify potential damages</li>
			<li>Color spaces is applied on sliding window to enhance the cropped sections</li>
			<li>During Training Crops which are saved earlier are used with above cmap feature + additional</li>
			<li>Enhanced colormap images are passed for feature creation and classification</li>
		</ul>

	</td>
  </tr>
</table>

### Solution 1
+--- solution1
|   +--- common
|   |   +--- constants.py
|   |   +--- geometry.py
|   |   +--- utils.py
|   |   +--- __init__.py
|   +--- damage_identification_engine.py   [Main File]
|   +--- feature_creation.py               [Feature creation / Manual User Annotation / Prediction]
|   +--- image
|   |   +--- operations.py
|   |   +--- preprocess_image.py           [Preprocess operation on image / Blob / Line(Angle/Extend/Brake/Repair)]
|   |   +--- __init__.py
|   +--- mlab.py                           [Model creation code]
|   +--- reshape_data_pixels.py
|   +--- sample.jpg                        [Image sample]
|   +--- sampleImages                      [Sample image directory]
|   |   +--- 16.jpg
|   |   +--- h1.jpg
|   |   +--- h2.jpg
|   |   +--- IMG1.jpg
|   |   +--- result.jpg
|   +--- sample_pred.jpg                   [Image sample prediction result]
|   +--- __init__.py

### Solution 2

+--- solution2
|   +--- extract_damage.py                 [Main File]
|   +--- generate_featureset.py            [Create Feature CSV using damage/undamage crops from disk / Subsampling]
|   +--- mlab.py                           [Model creation code]
|   +--- test.py
|   +--- utils
|   |   +--- common.py                     [Colormaps / Additional Features / Image resize]
|   |   +--- __init__.py
|   +--- __init__.py