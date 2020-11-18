from pyrouge import Rouge155

# r = Rouge155('pythonrouge/pythonrouge/RELEASE-1.5.5')
r = Rouge155()

r.system_dir = './outputs/Predicted/'
r.model_dir = './outputs/Reference/'
r.system_filename_pattern = '(\d+)_decoded.txt'
r.model_filename_pattern = '#ID#_reference.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)