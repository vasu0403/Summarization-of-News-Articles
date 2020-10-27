from pyrouge import Rouge155

r = Rouge155("/home/vasu/college/3rd_year/Information_Retrieval/pyrouge/tools/ROUGE-1.5.5")
r.system_dir = './predicted/'
r.model_dir = './target/'
r.system_filename_pattern = 'file.(\d+).txt'
r.model_filename_pattern = 'file.#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)