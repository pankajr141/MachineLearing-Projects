# Damage Identification
This repo contains code to identify damages on objects like rooftops due to hailsstorms.

## Solutions
<table>
  <tr>
    <td><b>Solution 1</b></td>
	<td>
	* Connected Components
	* Classifiers on Simple Features
	</td>
    <td>
		* The approach use Opencv to identify connected regions as potential damages
		* During Training Components are manually extracted by asking user Yes/No. 
		* During training Components marked as damages as saved in csv by extracting features.
		* Same features are used for classification during predictions
	</td>
  </tr>
  <tr>
    <td><b>Solution 2</b></td>
	<td>
	* Sliding Window
	* Color spaces
	* Classifier on Simple Features
	</td>
    <td>
	* Sliding Window of various sizes is used to identify potential damages
	* Color spaces is applied on sliding window to enhance the cropped sections
	* Enhanced colormap images are passed for feature creation and classification
	</td>
  </tr>
</table>