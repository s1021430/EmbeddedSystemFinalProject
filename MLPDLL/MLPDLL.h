#pragma once

#include "MLP.h"

using namespace std;

#ifdef MLPDLL_EXPORTS
#define MLPDLL_API __attribute__ ((visibility ("default")))
#else
#define MLPDLL_API __attribute__ ((visibility ("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif
	MLPDLL_API  unsigned long CreateMLP();
	MLPDLL_API	bool	DestroyMLP(unsigned long m_MLP);
	MLPDLL_API  bool	LoadNetwork(unsigned long m_MLP, char* network_load_file_name);
	MLPDLL_API  string  Classify(unsigned long m_MLP, BYTE* sample);
	MLPDLL_API  bool	SaveNetwork(unsigned long m_MLP, char* network_load_file_name);
	MLPDLL_API  bool	Training(unsigned long m_MLP, BYTE** samples, char *trainer_string, int num);
#ifdef __cplusplus
}
#endif