#include "MLPDLL.h"

MLPDLL_API unsigned long CreateMLP()
{
	return (unsigned long) new MLP();
}

MLPDLL_API bool DestroyMLP(unsigned long m_MLP)
{
	MLP* mlp = (MLP*)m_MLP;

	if (mlp) delete mlp;
	return true;
}

MLPDLL_API bool LoadNetwork(unsigned long m_MLP, char* network_load_file_name)
{
	MLP* mlp = (MLP*)m_MLP;
	return mlp->load_network(network_load_file_name);
}

MLPDLL_API string Classify(unsigned long m_MLP, BYTE* sample)
{
	MLP* mlp = (MLP*)m_MLP;
	return mlp->classify(sample);
}

MLPDLL_API bool SaveNetwork(unsigned long m_MLP, char* network_load_file_name)
{
	MLP* mlp = (MLP*)m_MLP;
	return mlp->save_network(network_load_file_name);
}

MLPDLL_API bool Training(unsigned long m_MLP, BYTE** samples, char* trainer_string, int num)
{
	MLP* mlp = (MLP*)m_MLP;
	return mlp->training(samples, trainer_string, num);
}