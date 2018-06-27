\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.09363 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.923829 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.16074 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.866709 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.921132 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.1445 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.2876 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.975609 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.863953 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.10099 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.975833 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.27125 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.0454 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.859539 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.2722 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.919134 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.01887 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.695971 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.9304 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.23311 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.952768 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.08175 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.835786 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.907613 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.895608 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.29252 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.679639 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.03005 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.46202 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.03503 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.963624 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.25902 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.32245 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.978744 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.11707 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.85882 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.743615 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.11144 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16628 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.852628 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.1923 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.887743 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.870678 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16214 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.0983 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.10218 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.641119 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.16229 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.767665 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.35538 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.56493 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.82043 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.27539 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.886893 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.753315 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.631172 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.697775 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.38439 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.811735 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.941442 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.748746 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.811232 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.11547 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.11271 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.59654 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.22833 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.64796 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.07055 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.18451 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.1909 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.62011 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.13856 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.53681 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.05301 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.8507 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.1379 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25181 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.814034 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.3253 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.778252 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.886809 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.3305 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.09227 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.847964 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.622996 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.932487 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.930044 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.4827 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.41406 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.723918 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.3989 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.888178 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.820283 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.486587 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.780269 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.27324 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.665681 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.957283 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.674614 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.760279 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.978628 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.995074 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.516535 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.39997 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.804 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.04805 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.10642 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.56005 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.74539 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.09132 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.45207 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.13518 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.725049 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.885095 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.11922 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.860722 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.20007 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.859852 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.947446 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.17439 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.16973 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.00607 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.775067 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.913454 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.936229 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.33431 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.1677 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.780159 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.33255 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.911486 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.99125 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.718267 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.833112 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.11341 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.744811 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.03392 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.890799 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.766438 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.01606 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.08797 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.689786 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.18354 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.55747 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.15178 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.901803 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.41545 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.32628 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.03877 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.31965 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.940621 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.757362 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.11501 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.12704 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.906278 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.12465 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.932751 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.85266 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.39962 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.659333 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.749997 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.564311 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.50567 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.805785 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.13652 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 2.06466 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.06871 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.906468 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.29296 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.426141 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.564941 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.765502 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.71665 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.0021 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.773492 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.629579 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.28731 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.981035 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.05082 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.649714 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.55794 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.16479 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.698071 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.71036 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.669454 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.3343 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.992947 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.971845 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.30183 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.920543 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.625309 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09996 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.861002 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07065 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.886013 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05687 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.17235 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.16845 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06316 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.835917 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.954817 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04099 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.27947 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.04471 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.899381 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.14911 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.835098 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.959689 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.761547 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.93515 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01701 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01534 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.00266 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.71375 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.741494 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.921472 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00399 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.705986 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.15834 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.35261 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13272 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.05978 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.23436 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.14217 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.99179 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.21903 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.999879 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.955616 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.002 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17321 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.838408 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.13273 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.85437 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.978393 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.42266 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.851213 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.803027 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.702623 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.14654 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.937033 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.26982 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.4335 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.999792 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.03968 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.06478 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.678097 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.637418 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.749519 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.46026 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.850704 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.90128 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.702788 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.00294 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.863659 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.984119 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.585587 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.30471 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.24719 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.938125 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.15548 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.929825 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.36518 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.940381 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.14235 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.0315 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.826952 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.714041 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.24867 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.897311 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12184 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.910418 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.842743 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.30861 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.634592 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.804599 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.647958 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.58243 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.696186 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.16168 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 2.26344 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.05619 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.819534 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.14497 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.513009 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.563117 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.596547 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.65733 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.803253 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.865755 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.604753 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.00017 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.08916 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.7931 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.438229 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.69486 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.42457 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.827785 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.84289 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.771723 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.706 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.12599 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.30398 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.10442 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.792246 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.834861 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14125 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.872548 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.07092 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.825493 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.976969 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.22903 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.20838 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.980652 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.809822 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.03591 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.0704 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.38745 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.07924 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.842215 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.19156 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.90284 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.006 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.711892 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.895617 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08352 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.769392 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.08075 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.802328 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.852932 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.952635 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.977392 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.713729 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.1445 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.38739 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.06389 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.958805 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.17628 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.19347 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.963683 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.12145 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.90063 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.795135 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.931365 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09044 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.90502 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.11187 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.945208 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.962816 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.15531 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.945717 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.951812 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.795163 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08397 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.939283 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.1622 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.27925 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.932935 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.09238 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.922649 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.854212 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.738977 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.845538 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11144 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.80916 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.912237 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.849171 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.910415 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.98517 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.03779 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.716311 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.15929 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.32606 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.949685 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07311 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.08979 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.21961 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.976388 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.2588 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.03774 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.82088 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.930263 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.172000309873677 z_k1_s0000
                                 - 0.266000837130325 z_k1_s0001
                                 - 0.101000063286253 z_k1_s0002
                                 - 0.416999460914296 z_k1_s0003
                                 - 0.115999971635667 z_k1_s0004
                                 - 0.347000860617312 z_k1_s0005
                                 - 0.101000063286253 z_k1_s0006
                                 - 0.100000036996282 z_k1_s0007
                                 - 0.6900026341004 z_k1_s0010
                                 - 0.780002354981173 z_k1_s0011
                                 - 0.328001051505089 z_k1_s0012
                                 - 0.214000450253215 z_k1_s0013
                                 - 0.900001982822204 z_k1_s0014
                                 - 0.589003583869463 z_k1_s0015
                                 - 0.72900155834584 z_k1_s0016
                                 - 0.544003723429076 z_k1_s0017
                                 - 0.479999551656917 z_k1_s0020
                                 - 0.368000472003 z_k1_s0021
                                 - 0.560998749487088 z_k1_s0022
                                 - 0.252999903889891 z_k1_s0023
                                 - 0.615999526592964 z_k1_s0024
                                 - 0.353999439922155 z_k1_s0025
                                 - 0.228999926307045 z_k1_s0026
                                 - 0.197999738583538 z_k1_s0027
                                 - 0.195000013329234 z_k1_s0030
                                 - 0.327000267718274 z_k1_s0031
                                 - 0.425999960110012 z_k1_s0032
                                 - 0.605000335283032 z_k1_s0033
                                 - 0.154000130653866 z_k1_s0034
                                 - 0.216000099109312 z_k1_s0035
                                 - 0.195000013329234 z_k1_s0036
                                 - 0.373999391702169 z_k1_s0037
                                 - 0.322000838102948 z_k1_s0040
                                 - 0.576002286225569 z_k1_s0041
                                 - 0.264000641735861 z_k1_s0042
                                 - 0.873001833338432 z_k1_s0043
                                 - 0.461000246399412 z_k1_s0044
                                 - 0.287000701600358 z_k1_s0045
                                 - 0.486999890297513 z_k1_s0046
                                 - 0.211000564326498 z_k1_s0047
                                 - 0.63000124984971 z_k1_s0050
                                 - 1.01099962906547 z_k1_s0051
                                 - 0.495000982024772 z_k1_s0052
                                 - 0.431000883349314 z_k1_s0053
                                 - 0.961000944541127 z_k1_s0054
                                 - 0.671003481487652 z_k1_s0055
                                 - 1.35500636631102 z_k1_s0056
                                 - 0.812004355377353 z_k1_s0057
                                 - 0.357999264249372 z_k1_s0060
                                 - 0.438999701683093 z_k1_s0061
                                 - 0.337000861137505 z_k1_s0062
                                 - 0.230999226236225 z_k1_s0063
                                 - 0.124000146197125 z_k1_s0064
                                 - 0.11300012307157 z_k1_s0065
                                 - 0.275000578138871 z_k1_s0066
                                 - 0.222000810188587 z_k1_s0067
                                 - 0.64200270956859 z_k1_s0070
                                 - 0.617001943137463 z_k1_s0071
                                 - 0.24400107850413 z_k1_s0072
                                 - 0.297001065652064 z_k1_s0073
                                 - 0.418000755403857 z_k1_s0074
                                 - 0.255001117972725 z_k1_s0075
                                 - 0.471002231357295 z_k1_s0076
                                 - 0.778001370373554 z_k1_s0077
                                 - 0.52599902922331 z_k1_s0100
                                 - 1.03599570288889 z_k1_s0101
                                 - 0.505998905270286 z_k1_s0102
                                 - 1.17799630090594 z_k1_s0103
                                 - 0.312999260395386 z_k1_s0104
                                 - 0.633996313976668 z_k1_s0105
                                 - 0.752998602768936 z_k1_s0106
                                 - 0.694997961255753 z_k1_s0107
                                 - 0.516999124197291 z_k1_s0110
                                 - 0.656999294035733 z_k1_s0111
                                 - 0.221999949529752 z_k1_s0112
                                 - 0.706000025028722 z_k1_s0113
                                 - 0.415999865091815 z_k1_s0114
                                 - 0.892001337941897 z_k1_s0115
                                 - 0.287999837725151 z_k1_s0116
                                 - 0.778997459367751 z_k1_s0117
                                 - 0.156000121183988 z_k1_s0120
                                 - 0.146999631116 z_k1_s0121
                                 - 0.353999598687188 z_k1_s0122
                                 - 0.129999682320266 z_k1_s0123
                                 - 0.20299961020773 z_k1_s0124
                                 - 0.0900000022831147 z_k1_s0125
                                 - 0.104999897997036 z_k1_s0126
                                 - 0.0710000004055692 z_k1_s0127
                                 - 0.327999904619597 z_k1_s0130
                                 - 0.11699987138755 z_k1_s0131
                                 - 0.239999975794792 z_k1_s0132
                                 - 0.0929998738080712 z_k1_s0133
                                 - 0.298000608519713 z_k1_s0134
                                 - 0.455000226673252 z_k1_s0135
                                 - 0.121999816370817 z_k1_s0136
                                 - 0.21600016511517 z_k1_s0137
                                 - 0.25499929857327 z_k1_s0140
                                 - 0.813998739512618 z_k1_s0141
                                 - 0.204999506678363 z_k1_s0142
                                 - 0.76499909541729 z_k1_s0143
                                 - 0.863998131508365 z_k1_s0144
                                 - 0.184999749880064 z_k1_s0145
                                 - 0.513998388546536 z_k1_s0146
                                 - 0.396999091458 z_k1_s0147
                                 - 0.742001256974981 z_k1_s0150
                                 - 0.539000516677761 z_k1_s0151
                                 - 0.82800061613862 z_k1_s0152
                                 - 1.06599870694022 z_k1_s0153
                                 - 0.891998239142722 z_k1_s0154
                                 - 0.639999574058174 z_k1_s0155
                                 - 0.543000514015625 z_k1_s0156
                                 - 0.585999026396587 z_k1_s0157
                                 - 0.168999926250811 z_k1_s0160
                                 - 0.171999926035736 z_k1_s0161
                                 - 0.25199998193367 z_k1_s0162
                                 - 0.164999803271023 z_k1_s0163
                                 - 0.183999925175434 z_k1_s0164
                                 - 0.28699967125807 z_k1_s0165
                                 - 0.466999658353549 z_k1_s0166
                                 - 0.152999804131324 z_k1_s0167
                                 - 0.232999659209013 z_k1_s0170
                                 - 0.909999986831231 z_k1_s0171
                                 - 1.07900006001404 z_k1_s0172
                                 - 0.392999418373153 z_k1_s0173
                                 - 0.861999370863165 z_k1_s0174
                                 - 0.287999460615634 z_k1_s0175
                                 - 1.08900040892368 z_k1_s0176
                                 - 0.43999947005116 z_k1_s0177
                                 - 0.210000068874462 z_k1_s1000
                                 - 0.24199895722289 z_k1_s1001
                                 - 0.317999623375588 z_k1_s1002
                                 - 0.18599997356309 z_k1_s1003
                                 - 0.118000067847557 z_k1_s1004
                                 - 0.297000906228212 z_k1_s1005
                                 - 0.565000148871766 z_k1_s1006
                                 - 0.249999717660054 z_k1_s1007
                                 - 0.898996660275773 z_k1_s1010
                                 - 0.333999454842336 z_k1_s1011
                                 - 1.17500062980416 z_k1_s1012
                                 - 0.311999717890736 z_k1_s1013
                                 - 0.921996991069805 z_k1_s1014
                                 - 0.37799944134484 z_k1_s1015
                                 - 0.354999623150112 z_k1_s1016
                                 - 0.750998989073343 z_k1_s1017
                                 - 0.638001735912851 z_k1_s1020
                                 - 0.987003675994219 z_k1_s1021
                                 - 1.44700545000788 z_k1_s1022
                                 - 0.629001280955046 z_k1_s1023
                                 - 0.517001544605575 z_k1_s1024
                                 - 0.300000722225251 z_k1_s1025
                                 - 1.39400395588145 z_k1_s1026
                                 - 0.753999748727802 z_k1_s1027
                                 - 0.293000652499562 z_k1_s1030
                                 - 0.13100006415371 z_k1_s1031
                                 - 0.345999628931128 z_k1_s1032
                                 - 0.117999991466695 z_k1_s1033
                                 - 0.321999026924292 z_k1_s1034
                                 - 0.385999184275734 z_k1_s1035
                                 - 0.0999999743616045 z_k1_s1036
                                 - 0.476000355801278 z_k1_s1037
                                 - 0.380000059665004 z_k1_s1040
                                 - 0.402000422013955 z_k1_s1041
                                 - 0.150999904077435 z_k1_s1042
                                 - 0.316999331983852 z_k1_s1043
                                 - 0.354999337950353 z_k1_s1044
                                 - 0.23200051495327 z_k1_s1045
                                 - 0.266000041765503 z_k1_s1046
                                 - 0.169999907060685 z_k1_s1047
                                 - 0.273000621138576 z_k1_s1050
                                 - 0.709000807933799 z_k1_s1051
                                 - 0.505000970266904 z_k1_s1052
                                 - 0.453000900869196 z_k1_s1053
                                 - 1.0570015884269 z_k1_s1054
                                 - 0.358000596299884 z_k1_s1055
                                 - 0.652001344233924 z_k1_s1056
                                 - 1.12900334384306 z_k1_s1057
                                 - 0.215000014491251 z_k1_s1060
                                 - 0.336000597656219 z_k1_s1061
                                 - 0.497000831470645 z_k1_s1062
                                 - 0.356000422980948 z_k1_s1063
                                 - 0.300999011087533 z_k1_s1064
                                 - 0.184000159087893 z_k1_s1065
                                 - 0.15399991650069 z_k1_s1066
                                 - 0.480000853794598 z_k1_s1067
                                 - 0.184000551595486 z_k1_s1070
                                 - 0.554002658872052 z_k1_s1071
                                 - 0.70100185193639 z_k1_s1072
                                 - 0.195000717649968 z_k1_s1073
                                 - 0.358001056578122 z_k1_s1074
                                 - 0.231000808935991 z_k1_s1075
                                 - 0.872001615992961 z_k1_s1076
                                 - 0.731001609173772 z_k1_s1077
                                 - 0.344998552170732 z_k1_s1100
                                 - 0.0809996543835878 z_k1_s1101
                                 - 0.118999498706725 z_k1_s1102
                                 - 0.0959995515955306 z_k1_s1103
                                 - 0.110999579706909 z_k1_s1104
                                 - 0.205999059615315 z_k1_s1105
                                 - 0.226999256050567 z_k1_s1106
                                 - 0.123999464444039 z_k1_s1107
                                 - 0.0490000305786162 z_k1_s1110
                                 - 0.130999638514387 z_k1_s1111
                                 - 0.226000227329919 z_k1_s1112
                                 - 0.0689999724556226 z_k1_s1113
                                 - 0.130000198800377 z_k1_s1114
                                 - 0.128000358372632 z_k1_s1115
                                 - 0.14200020236657 z_k1_s1116
                                 - 0.0859999614910671 z_k1_s1117
                                 - 0.0569999085944435 z_k1_s1120
                                 - 0.156000140184415 z_k1_s1121
                                 - 0.173999579783174 z_k1_s1122
                                 - 0.0799997886589483 z_k1_s1123
                                 - 0.0810002396917466 z_k1_s1124
                                 - 0.150999463018898 z_k1_s1125
                                 - 0.0519999415282401 z_k1_s1126
                                 - 0.0379999390629623 z_k1_s1127
                                 - 0.544000934065646 z_k1_s1130
                                 - 0.255000768937423 z_k1_s1131
                                 - 0.220999253744055 z_k1_s1132
                                 - 0.150000165627548 z_k1_s1133
                                 - 0.187000281353768 z_k1_s1134
                                 - 0.444000964914118 z_k1_s1135
                                 - 0.203000234037961 z_k1_s1136
                                 - 0.115000176424514 z_k1_s1137
                                 - 0.209999429877522 z_k1_s1140
                                 - 0.0699999628923528 z_k1_s1141
                                 - 0.148999737493908 z_k1_s1142
                                 - 0.0819998647711345 z_k1_s1143
                                 - 0.0929999277598633 z_k1_s1144
                                 - 0.113999870747615 z_k1_s1145
                                 - 0.314000108825419 z_k1_s1146
                                 - 0.115000397836973 z_k1_s1147
                                 - 0.185000339214096 z_k1_s1150
                                 - 0.0470000273625111 z_k1_s1151
                                 - 0.106000264068604 z_k1_s1152
                                 - 0.113000372752363 z_k1_s1153
                                 - 0.0960002239489272 z_k1_s1154
                                 - 0.0699999584377979 z_k1_s1155
                                 - 0.106999703880678 z_k1_s1156
                                 - 0.08200000658141 z_k1_s1157
                                 - 0.174999677150795 z_k1_s1160
                                 - 0.457998029562474 z_k1_s1161
                                 - 0.340999525713131 z_k1_s1162
                                 - 0.838997973101079 z_k1_s1163
                                 - 0.312999855604765 z_k1_s1164
                                 - 0.570998358737669 z_k1_s1165
                                 - 0.271999619535989 z_k1_s1166
                                 - 0.69199796605865 z_k1_s1167
                                 - 0.145999441952679 z_k1_s1170
                                 - 0.244998506271907 z_k1_s1171
                                 - 0.146999690161039 z_k1_s1172
                                 - 0.347999296421943 z_k1_s1173
                                 - 0.109999595619084 z_k1_s1174
                                 - 0.277999513773579 z_k1_s1175
                                 - 0.115999618874508 z_k1_s1176
                                 - 0.0759996593043144 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.982247 w_k1_ADM_s000
                                 + 1.30239 w_k1_AXP_s000
                                 + 1.05923 w_k1_DIS_s000
                                 + 1.13484 w_k1_FDX_s000
                                 + 0.877657 w_k1_IBM_s000
                                 + 0.939286 w_k1_INTC_s000
                                 + 1.01873 w_k1_KO_s000 + 1.53425 w_k1_MAT_s000
                                 + 1.05397 w_k1_MSFT_s000
                                 + 1.16417 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.609242 w_k1_ADM_s000
                                 + 1.02521 w_k1_AXP_s000
                                 + 0.96312 w_k1_DIS_s000
                                 + 0.85959 w_k1_FDX_s000
                                 + 1.54506 w_k1_IBM_s000
                                 + 1.08717 w_k1_INTC_s000
                                 + 1.14378 w_k1_KO_s000 + 1.90089 w_k1_MAT_s000
                                 + 0.916498 w_k1_MSFT_s000
                                 + 0.997208 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.912544 w_k1_ADM_s000
                                 + 1.0943 w_k1_AXP_s000 + 1.34306 w_k1_DIS_s000
                                 + 1.13592 w_k1_FDX_s000
                                 + 1.28166 w_k1_IBM_s000
                                 + 1.1173 w_k1_INTC_s000 + 1.35374 w_k1_KO_s000
                                 + 2.23541 w_k1_MAT_s000
                                 + 1.20229 w_k1_MSFT_s000
                                 + 1.35997 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.735369 w_k1_ADM_s000
                                 + 0.809783 w_k1_AXP_s000
                                 + 0.839161 w_k1_DIS_s000
                                 + 0.906578 w_k1_FDX_s000
                                 + 0.834021 w_k1_IBM_s000
                                 + 0.917795 w_k1_INTC_s000
                                 + 0.877141 w_k1_KO_s000
                                 + 0.961405 w_k1_MAT_s000
                                 + 0.934742 w_k1_MSFT_s000
                                 + 0.892014 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.13142 w_k1_ADM_s000
                                 + 1.48924 w_k1_AXP_s000 + 1.804 w_k1_DIS_s000
                                 + 1.38038 w_k1_FDX_s000
                                 + 1.12828 w_k1_IBM_s000
                                 + 1.04594 w_k1_INTC_s000
                                 + 1.48071 w_k1_KO_s000 + 1.33658 w_k1_MAT_s000
                                 + 1.27412 w_k1_MSFT_s000
                                 + 1.32752 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.892065 w_k1_ADM_s000
                                 + 1.00997 w_k1_AXP_s000
                                 + 0.731044 w_k1_DIS_s000
                                 + 0.938329 w_k1_FDX_s000
                                 + 0.921727 w_k1_IBM_s000
                                 + 0.942402 w_k1_INTC_s000
                                 + 0.983752 w_k1_KO_s000
                                 + 0.844123 w_k1_MAT_s000
                                 + 0.824901 w_k1_MSFT_s000
                                 + 1.03005 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.24712 w_k1_ADM_s000
                                 + 1.5165 w_k1_AXP_s000 + 1.20204 w_k1_DIS_s000
                                 + 1.21274 w_k1_FDX_s000
                                 + 1.43402 w_k1_IBM_s000
                                 + 1.11256 w_k1_INTC_s000
                                 + 1.38678 w_k1_KO_s000 + 1.76224 w_k1_MAT_s000
                                 + 1.16974 w_k1_MSFT_s000
                                 + 1.21735 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.911537 w_k1_ADM_s000
                                 + 0.894239 w_k1_AXP_s000
                                 + 1.19891 w_k1_DIS_s000
                                 + 1.01591 w_k1_FDX_s000
                                 + 1.30943 w_k1_IBM_s000
                                 + 0.931137 w_k1_INTC_s000
                                 + 1.19359 w_k1_KO_s000 + 1.27959 w_k1_MAT_s000
                                 + 1.08774 w_k1_MSFT_s000
                                 + 1.05602 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.90573 w_k1_ADM_s001
                                 + 0.812272 w_k1_AXP_s001
                                 + 0.771622 w_k1_DIS_s001
                                 + 0.923723 w_k1_FDX_s001
                                 + 0.685692 w_k1_IBM_s001
                                 + 1.0551 w_k1_INTC_s001
                                 + 0.904381 w_k1_KO_s001
                                 + 0.610876 w_k1_MAT_s001
                                 + 1.02226 w_k1_MSFT_s001
                                 + 0.903704 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.28343 w_k1_ADM_s001
                                 + 1.23212 w_k1_AXP_s001 + 1.2915 w_k1_DIS_s001
                                 + 1.12203 w_k1_FDX_s001
                                 + 1.49214 w_k1_IBM_s001
                                 + 0.911332 w_k1_INTC_s001
                                 + 1.28382 w_k1_KO_s001 + 1.79632 w_k1_MAT_s001
                                 + 1.10605 w_k1_MSFT_s001
                                 + 1.22612 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.885504 w_k1_ADM_s001
                                 + 0.740196 w_k1_AXP_s001
                                 + 0.748096 w_k1_DIS_s001
                                 + 0.858961 w_k1_FDX_s001
                                 + 0.952139 w_k1_IBM_s001
                                 + 0.906975 w_k1_INTC_s001
                                 + 0.898027 w_k1_KO_s001
                                 + 0.987887 w_k1_MAT_s001
                                 + 0.846297 w_k1_MSFT_s001
                                 + 0.910901 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.45792 w_k1_ADM_s001
                                 + 1.61936 w_k1_AXP_s001
                                 + 1.64801 w_k1_DIS_s001
                                 + 1.38284 w_k1_FDX_s001
                                 + 1.38146 w_k1_IBM_s001
                                 + 1.42091 w_k1_INTC_s001
                                 + 1.48844 w_k1_KO_s001 + 1.43097 w_k1_MAT_s001
                                 + 1.41091 w_k1_MSFT_s001
                                 + 1.34661 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.28175 w_k1_ADM_s001
                                 + 1.02445 w_k1_AXP_s001
                                 + 0.932429 w_k1_DIS_s001
                                 + 1.09788 w_k1_FDX_s001
                                 + 0.971817 w_k1_IBM_s001
                                 + 0.927575 w_k1_INTC_s001
                                 + 1.0858 w_k1_KO_s001 + 1.02444 w_k1_MAT_s001
                                 + 0.950098 w_k1_MSFT_s001
                                 + 1.03226 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.2469 w_k1_ADM_s001
                                 + 0.918733 w_k1_AXP_s001
                                 + 1.14718 w_k1_DIS_s001
                                 + 1.23823 w_k1_FDX_s001
                                 + 0.785168 w_k1_IBM_s001
                                 + 0.953292 w_k1_INTC_s001
                                 + 1.00425 w_k1_KO_s001
                                 + 0.629041 w_k1_MAT_s001
                                 + 1.1451 w_k1_MSFT_s001 + 1.02764 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.843348 w_k1_ADM_s001
                                 + 0.880305 w_k1_AXP_s001
                                 + 0.961851 w_k1_DIS_s001
                                 + 0.878865 w_k1_FDX_s001
                                 + 1.12147 w_k1_IBM_s001
                                 + 0.854714 w_k1_INTC_s001
                                 + 1.10547 w_k1_KO_s001 + 1.49733 w_k1_MAT_s001
                                 + 0.897158 w_k1_MSFT_s001
                                 + 1.02107 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.74685 w_k1_ADM_s001
                                 + 0.67528 w_k1_AXP_s001
                                 + 0.575112 w_k1_DIS_s001
                                 + 0.746835 w_k1_FDX_s001
                                 + 0.816987 w_k1_IBM_s001
                                 + 0.663517 w_k1_INTC_s001
                                 + 0.684754 w_k1_KO_s001
                                 + 0.836093 w_k1_MAT_s001
                                 + 0.682839 w_k1_MSFT_s001
                                 + 0.813254 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.665259 w_k1_ADM_s002
                                 + 0.847328 w_k1_AXP_s002
                                 + 0.720633 w_k1_DIS_s002
                                 + 0.770121 w_k1_FDX_s002
                                 + 0.929445 w_k1_IBM_s002
                                 + 0.85863 w_k1_INTC_s002
                                 + 0.791822 w_k1_KO_s002
                                 + 0.819066 w_k1_MAT_s002
                                 + 0.819909 w_k1_MSFT_s002
                                 + 0.842989 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.857509 w_k1_ADM_s002
                                 + 1.51872 w_k1_AXP_s002
                                 + 1.36883 w_k1_DIS_s002
                                 + 1.12117 w_k1_FDX_s002
                                 + 1.61228 w_k1_IBM_s002
                                 + 1.16575 w_k1_INTC_s002
                                 + 1.23526 w_k1_KO_s002 + 1.35582 w_k1_MAT_s002
                                 + 1.15271 w_k1_MSFT_s002
                                 + 1.22241 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.831495 w_k1_ADM_s002
                                 + 1.12713 w_k1_AXP_s002
                                 + 0.758704 w_k1_DIS_s002
                                 + 0.937829 w_k1_FDX_s002
                                 + 1.15968 w_k1_IBM_s002
                                 + 1.08479 w_k1_INTC_s002
                                 + 0.882841 w_k1_KO_s002
                                 + 0.78193 w_k1_MAT_s002
                                 + 0.937901 w_k1_MSFT_s002
                                 + 0.983254 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.13665 w_k1_ADM_s002
                                 + 1.42119 w_k1_AXP_s002
                                 + 1.62785 w_k1_DIS_s002
                                 + 1.42548 w_k1_FDX_s002
                                 + 1.12237 w_k1_IBM_s002
                                 + 1.5546 w_k1_INTC_s002 + 1.1681 w_k1_KO_s002
                                 + 0.605346 w_k1_MAT_s002
                                 + 1.49161 w_k1_MSFT_s002
                                 + 1.15032 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.892184 w_k1_ADM_s002
                                 + 0.918508 w_k1_AXP_s002
                                 + 1.11529 w_k1_DIS_s002
                                 + 0.973845 w_k1_FDX_s002
                                 + 1.13896 w_k1_IBM_s002
                                 + 1.02676 w_k1_INTC_s002
                                 + 0.920721 w_k1_KO_s002
                                 + 0.71174 w_k1_MAT_s002
                                 + 1.02874 w_k1_MSFT_s002
                                 + 0.995702 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.11092 w_k1_ADM_s002
                                 + 1.11857 w_k1_AXP_s002
                                 + 1.00955 w_k1_DIS_s002
                                 + 1.16328 w_k1_FDX_s002
                                 + 0.808773 w_k1_IBM_s002
                                 + 1.19797 w_k1_INTC_s002
                                 + 0.867308 w_k1_KO_s002
                                 + 0.619144 w_k1_MAT_s002
                                 + 1.16932 w_k1_MSFT_s002
                                 + 1.07931 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.676548 w_k1_ADM_s002
                                 + 0.623944 w_k1_AXP_s002
                                 + 0.498896 w_k1_DIS_s002
                                 + 0.655862 w_k1_FDX_s002
                                 + 0.804943 w_k1_IBM_s002
                                 + 0.718192 w_k1_INTC_s002
                                 + 0.651712 w_k1_KO_s002
                                 + 0.673611 w_k1_MAT_s002
                                 + 0.767595 w_k1_MSFT_s002
                                 + 0.739767 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.971297 w_k1_ADM_s002
                                 + 0.953397 w_k1_AXP_s002
                                 + 1.02077 w_k1_DIS_s002
                                 + 1.07653 w_k1_FDX_s002
                                 + 0.677849 w_k1_IBM_s002
                                 + 1.041 w_k1_INTC_s002 + 0.874059 w_k1_KO_s002
                                 + 0.570234 w_k1_MAT_s002
                                 + 1.21077 w_k1_MSFT_s002
                                 + 1.02261 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.01247 w_k1_ADM_s003
                                 + 1.00812 w_k1_AXP_s003
                                 + 1.14469 w_k1_DIS_s003
                                 + 1.04187 w_k1_FDX_s003
                                 + 0.923222 w_k1_IBM_s003
                                 + 1.24405 w_k1_INTC_s003
                                 + 0.994784 w_k1_KO_s003
                                 + 0.862246 w_k1_MAT_s003
                                 + 1.09726 w_k1_MSFT_s003
                                 + 1.07253 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.835369 w_k1_ADM_s003
                                 + 0.758887 w_k1_AXP_s003
                                 + 0.923421 w_k1_DIS_s003
                                 + 1.03167 w_k1_FDX_s003
                                 + 0.736603 w_k1_IBM_s003
                                 + 1.18816 w_k1_INTC_s003
                                 + 0.840376 w_k1_KO_s003
                                 + 0.531146 w_k1_MAT_s003
                                 + 0.988153 w_k1_MSFT_s003
                                 + 0.859412 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.741619 w_k1_ADM_s003
                                 + 0.658999 w_k1_AXP_s003
                                 + 0.862501 w_k1_DIS_s003
                                 + 0.813901 w_k1_FDX_s003
                                 + 1.00648 w_k1_IBM_s003
                                 + 1.03202 w_k1_INTC_s003
                                 + 0.994843 w_k1_KO_s003
                                 + 0.650398 w_k1_MAT_s003
                                 + 0.765854 w_k1_MSFT_s003
                                 + 0.858903 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.949908 w_k1_ADM_s003
                                 + 1.2269 w_k1_AXP_s003 + 1.13326 w_k1_DIS_s003
                                 + 1.04481 w_k1_FDX_s003 + 1.298 w_k1_IBM_s003
                                 + 1.17061 w_k1_INTC_s003
                                 + 1.10167 w_k1_KO_s003 + 1.22541 w_k1_MAT_s003
                                 + 0.975403 w_k1_MSFT_s003
                                 + 0.926897 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.586879 w_k1_ADM_s003
                                 + 0.858762 w_k1_AXP_s003
                                 + 0.767788 w_k1_DIS_s003
                                 + 0.847425 w_k1_FDX_s003
                                 + 1.0558 w_k1_IBM_s003
                                 + 0.77623 w_k1_INTC_s003
                                 + 0.974303 w_k1_KO_s003
                                 + 1.34028 w_k1_MAT_s003
                                 + 0.724917 w_k1_MSFT_s003
                                 + 0.925992 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.560286 w_k1_ADM_s003
                                 + 0.585773 w_k1_AXP_s003
                                 + 0.60429 w_k1_DIS_s003
                                 + 0.761365 w_k1_FDX_s003
                                 + 0.733655 w_k1_IBM_s003
                                 + 0.935811 w_k1_INTC_s003
                                 + 0.811583 w_k1_KO_s003
                                 + 0.699163 w_k1_MAT_s003
                                 + 0.740797 w_k1_MSFT_s003
                                 + 0.750737 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.10008 w_k1_ADM_s003
                                 + 1.26001 w_k1_AXP_s003
                                 + 1.05419 w_k1_DIS_s003
                                 + 1.27345 w_k1_FDX_s003
                                 + 0.779565 w_k1_IBM_s003
                                 + 1.48935 w_k1_INTC_s003
                                 + 0.998453 w_k1_KO_s003 + 0.7717 w_k1_MAT_s003
                                 + 1.13113 w_k1_MSFT_s003
                                 + 1.04831 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.22892 w_k1_ADM_s003
                                 + 1.17355 w_k1_AXP_s003
                                 + 1.57213 w_k1_DIS_s003
                                 + 1.27484 w_k1_FDX_s003 + 1.0745 w_k1_IBM_s003
                                 + 1.12039 w_k1_INTC_s003
                                 + 1.29798 w_k1_KO_s003 + 1.19246 w_k1_MAT_s003
                                 + 1.11108 w_k1_MSFT_s003
                                 + 1.08747 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.831662 w_k1_ADM_s004
                                 + 0.706763 w_k1_AXP_s004
                                 + 0.619878 w_k1_DIS_s004
                                 + 0.716961 w_k1_FDX_s004
                                 + 0.76591 w_k1_IBM_s004
                                 + 0.775668 w_k1_INTC_s004
                                 + 0.829586 w_k1_KO_s004
                                 + 0.938717 w_k1_MAT_s004
                                 + 0.773195 w_k1_MSFT_s004
                                 + 0.880531 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.999138 w_k1_ADM_s004
                                 + 0.818784 w_k1_AXP_s004
                                 + 0.804072 w_k1_DIS_s004
                                 + 0.916555 w_k1_FDX_s004
                                 + 0.750619 w_k1_IBM_s004
                                 + 0.83257 w_k1_INTC_s004
                                 + 0.762626 w_k1_KO_s004
                                 + 0.839051 w_k1_MAT_s004
                                 + 0.854263 w_k1_MSFT_s004
                                 + 0.925683 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.768181 w_k1_ADM_s004
                                 + 1.03197 w_k1_AXP_s004
                                 + 0.830375 w_k1_DIS_s004
                                 + 0.83602 w_k1_FDX_s004
                                 + 1.20885 w_k1_IBM_s004
                                 + 1.00176 w_k1_INTC_s004
                                 + 1.03305 w_k1_KO_s004 + 1.46049 w_k1_MAT_s004
                                 + 0.898811 w_k1_MSFT_s004
                                 + 1.06588 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.08701 w_k1_ADM_s004
                                 + 1.35795 w_k1_AXP_s004
                                 + 1.37633 w_k1_DIS_s004 + 1.2319 w_k1_FDX_s004
                                 + 1.33701 w_k1_IBM_s004
                                 + 1.14968 w_k1_INTC_s004
                                 + 1.26489 w_k1_KO_s004 + 1.75697 w_k1_MAT_s004
                                 + 1.15052 w_k1_MSFT_s004
                                 + 1.20331 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09694 w_k1_ADM_s004
                                 + 1.147 w_k1_AXP_s004 + 1.32308 w_k1_DIS_s004
                                 + 1.1502 w_k1_FDX_s004
                                 + 0.914682 w_k1_IBM_s004
                                 + 0.964391 w_k1_INTC_s004
                                 + 1.03956 w_k1_KO_s004 + 1.00098 w_k1_MAT_s004
                                 + 1.16429 w_k1_MSFT_s004
                                 + 1.02133 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.975455 w_k1_ADM_s004
                                 + 0.981179 w_k1_AXP_s004
                                 + 1.0236 w_k1_DIS_s004
                                 + 0.881944 w_k1_FDX_s004
                                 + 1.61193 w_k1_IBM_s004
                                 + 0.880765 w_k1_INTC_s004
                                 + 1.0798 w_k1_KO_s004 + 1.38398 w_k1_MAT_s004
                                 + 0.844945 w_k1_MSFT_s004
                                 + 1.00942 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.96137 w_k1_ADM_s004
                                 + 1.13055 w_k1_AXP_s004
                                 + 0.991947 w_k1_DIS_s004
                                 + 1.00956 w_k1_FDX_s004
                                 + 0.883674 w_k1_IBM_s004
                                 + 0.953978 w_k1_INTC_s004
                                 + 0.972089 w_k1_KO_s004
                                 + 1.03469 w_k1_MAT_s004
                                 + 1.01216 w_k1_MSFT_s004
                                 + 1.05122 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.811649 w_k1_ADM_s004
                                 + 0.793958 w_k1_AXP_s004
                                 + 1.16053 w_k1_DIS_s004
                                 + 0.947165 w_k1_FDX_s004
                                 + 1.13083 w_k1_IBM_s004
                                 + 1.07736 w_k1_INTC_s004
                                 + 1.15102 w_k1_KO_s004 + 0.95562 w_k1_MAT_s004
                                 + 1.0296 w_k1_MSFT_s004 + 1.05179 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.10646 w_k1_ADM_s005
                                 + 1.24993 w_k1_AXP_s005
                                 + 0.952033 w_k1_DIS_s005
                                 + 1.10873 w_k1_FDX_s005
                                 + 0.793847 w_k1_IBM_s005
                                 + 0.91434 w_k1_INTC_s005
                                 + 0.945031 w_k1_KO_s005
                                 + 1.09695 w_k1_MAT_s005
                                 + 0.894396 w_k1_MSFT_s005
                                 + 0.941941 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.761782 w_k1_ADM_s005
                                 + 0.89479 w_k1_AXP_s005
                                 + 0.81634 w_k1_DIS_s005
                                 + 0.881019 w_k1_FDX_s005
                                 + 0.749866 w_k1_IBM_s005
                                 + 0.874378 w_k1_INTC_s005
                                 + 0.760971 w_k1_KO_s005
                                 + 0.758407 w_k1_MAT_s005
                                 + 0.884898 w_k1_MSFT_s005
                                 + 0.935073 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.23324 w_k1_ADM_s005
                                 + 1.14723 w_k1_AXP_s005
                                 + 1.21327 w_k1_DIS_s005
                                 + 1.28257 w_k1_FDX_s005
                                 + 0.734239 w_k1_IBM_s005
                                 + 1.42971 w_k1_INTC_s005
                                 + 0.952278 w_k1_KO_s005
                                 + 0.970394 w_k1_MAT_s005
                                 + 1.25615 w_k1_MSFT_s005
                                 + 1.21212 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.71683 w_k1_ADM_s005
                                 + 1.35831 w_k1_AXP_s005
                                 + 0.960688 w_k1_DIS_s005
                                 + 0.929126 w_k1_FDX_s005
                                 + 1.48106 w_k1_IBM_s005
                                 + 0.992173 w_k1_INTC_s005
                                 + 1.13023 w_k1_KO_s005 + 1.83199 w_k1_MAT_s005
                                 + 0.901667 w_k1_MSFT_s005
                                 + 1.04295 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.02087 w_k1_ADM_s005
                                 + 0.949106 w_k1_AXP_s005
                                 + 0.896181 w_k1_DIS_s005
                                 + 0.911722 w_k1_FDX_s005
                                 + 1.06735 w_k1_IBM_s005
                                 + 0.998461 w_k1_INTC_s005
                                 + 0.971657 w_k1_KO_s005
                                 + 0.763446 w_k1_MAT_s005
                                 + 0.857991 w_k1_MSFT_s005
                                 + 0.993937 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.08002 w_k1_ADM_s005
                                 + 1.05951 w_k1_AXP_s005
                                 + 1.23512 w_k1_DIS_s005
                                 + 1.06502 w_k1_FDX_s005
                                 + 1.32503 w_k1_IBM_s005
                                 + 1.15635 w_k1_INTC_s005
                                 + 1.24079 w_k1_KO_s005 + 1.03554 w_k1_MAT_s005
                                 + 1.07234 w_k1_MSFT_s005
                                 + 1.13171 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.804418 w_k1_ADM_s005
                                 + 0.783886 w_k1_AXP_s005
                                 + 0.62492 w_k1_DIS_s005
                                 + 0.705562 w_k1_FDX_s005
                                 + 0.855371 w_k1_IBM_s005
                                 + 0.87281 w_k1_INTC_s005
                                 + 0.66837 w_k1_KO_s005
                                 + 0.651697 w_k1_MAT_s005
                                 + 0.745099 w_k1_MSFT_s005
                                 + 0.812764 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.42427 w_k1_ADM_s005
                                 + 1.75623 w_k1_AXP_s005
                                 + 1.62161 w_k1_DIS_s005 + 1.4248 w_k1_FDX_s005
                                 + 1.26656 w_k1_IBM_s005
                                 + 1.35347 w_k1_INTC_s005
                                 + 1.30413 w_k1_KO_s005 + 1.42511 w_k1_MAT_s005
                                 + 1.28648 w_k1_MSFT_s005
                                 + 1.20203 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.40029 w_k1_ADM_s006
                                 + 0.946819 w_k1_AXP_s006
                                 + 0.98173 w_k1_DIS_s006
                                 + 1.01108 w_k1_FDX_s006
                                 + 0.863728 w_k1_IBM_s006
                                 + 1.18626 w_k1_INTC_s006
                                 + 0.972163 w_k1_KO_s006
                                 + 0.52612 w_k1_MAT_s006
                                 + 1.15661 w_k1_MSFT_s006
                                 + 0.933732 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.870705 w_k1_ADM_s006
                                 + 1.02189 w_k1_AXP_s006
                                 + 0.956706 w_k1_DIS_s006
                                 + 0.919657 w_k1_FDX_s006
                                 + 1.12794 w_k1_IBM_s006
                                 + 1.1405 w_k1_INTC_s006 + 1.02101 w_k1_KO_s006
                                 + 0.982859 w_k1_MAT_s006
                                 + 0.932368 w_k1_MSFT_s006
                                 + 0.872135 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.957527 w_k1_ADM_s006
                                 + 0.848754 w_k1_AXP_s006
                                 + 0.719009 w_k1_DIS_s006
                                 + 0.888062 w_k1_FDX_s006
                                 + 0.737012 w_k1_IBM_s006
                                 + 1.16379 w_k1_INTC_s006
                                 + 0.909831 w_k1_KO_s006
                                 + 0.700993 w_k1_MAT_s006
                                 + 0.955555 w_k1_MSFT_s006
                                 + 0.788138 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.40545 w_k1_ADM_s006
                                 + 1.43804 w_k1_AXP_s006
                                 + 1.54258 w_k1_DIS_s006
                                 + 1.34702 w_k1_FDX_s006
                                 + 1.06124 w_k1_IBM_s006 + 1.298 w_k1_INTC_s006
                                 + 1.3314 w_k1_KO_s006 + 1.4206 w_k1_MAT_s006
                                 + 1.51958 w_k1_MSFT_s006 + 1.1583 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.38318 w_k1_ADM_s006
                                 + 1.18019 w_k1_AXP_s006
                                 + 1.24325 w_k1_DIS_s006
                                 + 1.28935 w_k1_FDX_s006
                                 + 0.758015 w_k1_IBM_s006
                                 + 1.28393 w_k1_INTC_s006
                                 + 1.20952 w_k1_KO_s006
                                 + 0.894165 w_k1_MAT_s006
                                 + 1.35282 w_k1_MSFT_s006
                                 + 1.07138 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.34809 w_k1_ADM_s006
                                 + 1.34063 w_k1_AXP_s006
                                 + 1.83797 w_k1_DIS_s006
                                 + 1.31832 w_k1_FDX_s006
                                 + 1.31905 w_k1_IBM_s006
                                 + 1.70495 w_k1_INTC_s006
                                 + 1.47161 w_k1_KO_s006 + 1.36761 w_k1_MAT_s006
                                 + 1.6465 w_k1_MSFT_s006 + 1.17838 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.945235 w_k1_ADM_s006
                                 + 1.09026 w_k1_AXP_s006
                                 + 1.19282 w_k1_DIS_s006
                                 + 0.926975 w_k1_FDX_s006
                                 + 1.62968 w_k1_IBM_s006
                                 + 1.13971 w_k1_INTC_s006
                                 + 1.43413 w_k1_KO_s006 + 1.38031 w_k1_MAT_s006
                                 + 1.07185 w_k1_MSFT_s006
                                 + 1.00942 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.907293 w_k1_ADM_s006
                                 + 0.575927 w_k1_AXP_s006
                                 + 0.580417 w_k1_DIS_s006
                                 + 0.712773 w_k1_FDX_s006
                                 + 0.747903 w_k1_IBM_s006
                                 + 1.02872 w_k1_INTC_s006
                                 + 0.858856 w_k1_KO_s006
                                 + 0.497818 w_k1_MAT_s006
                                 + 0.933486 w_k1_MSFT_s006
                                 + 0.802286 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.966879 w_k1_ADM_s007
                                 + 1.04488 w_k1_AXP_s007
                                 + 0.957877 w_k1_DIS_s007
                                 + 0.990682 w_k1_FDX_s007
                                 + 1.18878 w_k1_IBM_s007
                                 + 0.876361 w_k1_INTC_s007
                                 + 0.936158 w_k1_KO_s007
                                 + 1.00813 w_k1_MAT_s007
                                 + 0.945446 w_k1_MSFT_s007
                                 + 1.02282 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.880707 w_k1_ADM_s007
                                 + 0.763289 w_k1_AXP_s007
                                 + 0.956436 w_k1_DIS_s007
                                 + 0.87848 w_k1_FDX_s007
                                 + 1.14063 w_k1_IBM_s007
                                 + 0.96688 w_k1_INTC_s007
                                 + 0.921113 w_k1_KO_s007
                                 + 0.874303 w_k1_MAT_s007
                                 + 1.03505 w_k1_MSFT_s007
                                 + 0.958333 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.991347 w_k1_ADM_s007
                                 + 0.796422 w_k1_AXP_s007
                                 + 0.7894 w_k1_DIS_s007 + 0.94278 w_k1_FDX_s007
                                 + 0.648545 w_k1_IBM_s007
                                 + 0.90889 w_k1_INTC_s007
                                 + 0.70124 w_k1_KO_s007
                                 + 0.520806 w_k1_MAT_s007
                                 + 0.911831 w_k1_MSFT_s007
                                 + 0.860672 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.00792 w_k1_ADM_s007
                                 + 1.17832 w_k1_AXP_s007
                                 + 1.51646 w_k1_DIS_s007
                                 + 1.21401 w_k1_FDX_s007
                                 + 1.31899 w_k1_IBM_s007
                                 + 1.26521 w_k1_INTC_s007
                                 + 1.10855 w_k1_KO_s007 + 1.24465 w_k1_MAT_s007
                                 + 1.37744 w_k1_MSFT_s007
                                 + 1.11269 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.689665 w_k1_ADM_s007
                                 + 0.628101 w_k1_AXP_s007
                                 + 0.676336 w_k1_DIS_s007
                                 + 0.743822 w_k1_FDX_s007
                                 + 0.808266 w_k1_IBM_s007
                                 + 0.735917 w_k1_INTC_s007
                                 + 0.754387 w_k1_KO_s007
                                 + 0.488383 w_k1_MAT_s007
                                 + 0.696227 w_k1_MSFT_s007
                                 + 0.798337 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.23155 w_k1_ADM_s007
                                 + 1.40222 w_k1_AXP_s007
                                 + 1.67491 w_k1_DIS_s007
                                 + 1.26753 w_k1_FDX_s007
                                 + 1.53848 w_k1_IBM_s007
                                 + 1.27909 w_k1_INTC_s007
                                 + 1.26613 w_k1_KO_s007 + 1.24263 w_k1_MAT_s007
                                 + 1.30764 w_k1_MSFT_s007
                                 + 1.22469 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.863941 w_k1_ADM_s007
                                 + 1.04255 w_k1_AXP_s007
                                 + 0.801862 w_k1_DIS_s007
                                 + 0.967088 w_k1_FDX_s007
                                 + 0.702596 w_k1_IBM_s007
                                 + 0.989655 w_k1_INTC_s007
                                 + 0.815921 w_k1_KO_s007
                                 + 0.728582 w_k1_MAT_s007
                                 + 0.972586 w_k1_MSFT_s007
                                 + 0.961632 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.20198 w_k1_ADM_s007
                                 + 1.43651 w_k1_AXP_s007
                                 + 1.19215 w_k1_DIS_s007
                                 + 1.25925 w_k1_FDX_s007
                                 + 1.08067 w_k1_IBM_s007
                                 + 1.1059 w_k1_INTC_s007 + 1.02241 w_k1_KO_s007
                                 + 1.21494 w_k1_MAT_s007
                                 + 1.25078 w_k1_MSFT_s007
                                 + 1.10292 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.53056 w_k1_ADM_s010
                                 + 1.51986 w_k1_AXP_s010
                                 + 1.54827 w_k1_DIS_s010
                                 + 1.30412 w_k1_FDX_s010
                                 + 1.56583 w_k1_IBM_s010
                                 + 1.15087 w_k1_INTC_s010
                                 + 1.64484 w_k1_KO_s010 + 2.18737 w_k1_MAT_s010
                                 + 1.32445 w_k1_MSFT_s010
                                 + 1.31484 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.08092 w_k1_ADM_s010
                                 + 0.930275 w_k1_AXP_s010
                                 + 0.936342 w_k1_DIS_s010
                                 + 1.04475 w_k1_FDX_s010
                                 + 0.764846 w_k1_IBM_s010
                                 + 0.991086 w_k1_INTC_s010
                                 + 0.863028 w_k1_KO_s010
                                 + 0.694347 w_k1_MAT_s010
                                 + 1.06264 w_k1_MSFT_s010 + 0.9351 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.767764 w_k1_ADM_s010
                                 + 0.859093 w_k1_AXP_s010
                                 + 0.762981 w_k1_DIS_s010
                                 + 0.778767 w_k1_FDX_s010
                                 + 1.29577 w_k1_IBM_s010
                                 + 0.837714 w_k1_INTC_s010
                                 + 1.07735 w_k1_KO_s010 + 1.60813 w_k1_MAT_s010
                                 + 0.939308 w_k1_MSFT_s010
                                 + 0.986187 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.819037 w_k1_ADM_s010
                                 + 0.777878 w_k1_AXP_s010
                                 + 0.653413 w_k1_DIS_s010
                                 + 0.800783 w_k1_FDX_s010
                                 + 0.810286 w_k1_IBM_s010
                                 + 0.915096 w_k1_INTC_s010
                                 + 0.811674 w_k1_KO_s010
                                 + 0.672483 w_k1_MAT_s010
                                 + 0.811572 w_k1_MSFT_s010
                                 + 0.755746 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.25876 w_k1_ADM_s010
                                 + 0.918064 w_k1_AXP_s010
                                 + 1.0101 w_k1_DIS_s010 + 1.05905 w_k1_FDX_s010
                                 + 1.13415 w_k1_IBM_s010
                                 + 1.19416 w_k1_INTC_s010
                                 + 1.18326 w_k1_KO_s010 + 1.09754 w_k1_MAT_s010
                                 + 1.08869 w_k1_MSFT_s010
                                 + 1.06769 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.0192 w_k1_ADM_s010
                                 + 1.3151 w_k1_AXP_s010
                                 + 0.979451 w_k1_DIS_s010
                                 + 0.992429 w_k1_FDX_s010
                                 + 1.47606 w_k1_IBM_s010
                                 + 1.25687 w_k1_INTC_s010
                                 + 1.27915 w_k1_KO_s010 + 2.33954 w_k1_MAT_s010
                                 + 1.102 w_k1_MSFT_s010 + 1.2219 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.0379 w_k1_ADM_s010
                                 + 1.15342 w_k1_AXP_s010
                                 + 1.30108 w_k1_DIS_s010
                                 + 1.25686 w_k1_FDX_s010
                                 + 1.02238 w_k1_IBM_s010
                                 + 1.32558 w_k1_INTC_s010 + 1.1965 w_k1_KO_s010
                                 + 1.15466 w_k1_MAT_s010
                                 + 1.30056 w_k1_MSFT_s010 + 1.1407 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.749347 w_k1_ADM_s010
                                 + 0.876286 w_k1_AXP_s010
                                 + 0.911927 w_k1_DIS_s010
                                 + 0.889824 w_k1_FDX_s010
                                 + 1.14836 w_k1_IBM_s010
                                 + 1.0655 w_k1_INTC_s010 + 1.12504 w_k1_KO_s010
                                 + 1.29303 w_k1_MAT_s010
                                 + 1.02082 w_k1_MSFT_s010
                                 + 1.05993 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.817618 w_k1_ADM_s011
                                 + 1.13398 w_k1_AXP_s011
                                 + 0.842818 w_k1_DIS_s011
                                 + 0.859033 w_k1_FDX_s011
                                 + 1.36142 w_k1_IBM_s011
                                 + 1.29928 w_k1_INTC_s011
                                 + 1.02995 w_k1_KO_s011 + 1.33842 w_k1_MAT_s011
                                 + 0.94543 w_k1_MSFT_s011
                                 + 0.968441 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.750676 w_k1_ADM_s011
                                 + 0.816252 w_k1_AXP_s011
                                 + 0.906584 w_k1_DIS_s011
                                 + 0.880736 w_k1_FDX_s011
                                 + 0.673709 w_k1_IBM_s011
                                 + 1.11235 w_k1_INTC_s011
                                 + 0.855395 w_k1_KO_s011
                                 + 0.764349 w_k1_MAT_s011
                                 + 1.12559 w_k1_MSFT_s011
                                 + 0.894821 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.629275 w_k1_ADM_s011
                                 + 0.604809 w_k1_AXP_s011
                                 + 0.536882 w_k1_DIS_s011
                                 + 0.609671 w_k1_FDX_s011
                                 + 0.974198 w_k1_IBM_s011
                                 + 0.917314 w_k1_INTC_s011
                                 + 0.785714 w_k1_KO_s011
                                 + 0.745462 w_k1_MAT_s011
                                 + 0.792595 w_k1_MSFT_s011
                                 + 0.892433 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.89417 w_k1_ADM_s011
                                 + 0.903677 w_k1_AXP_s011
                                 + 0.818404 w_k1_DIS_s011
                                 + 0.830693 w_k1_FDX_s011
                                 + 0.844361 w_k1_IBM_s011
                                 + 1.13509 w_k1_INTC_s011
                                 + 0.945946 w_k1_KO_s011
                                 + 1.03488 w_k1_MAT_s011
                                 + 0.999853 w_k1_MSFT_s011
                                 + 0.944476 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.31793 w_k1_ADM_s011
                                 + 1.60687 w_k1_AXP_s011
                                 + 1.41664 w_k1_DIS_s011
                                 + 1.18427 w_k1_FDX_s011
                                 + 1.61572 w_k1_IBM_s011
                                 + 1.3053 w_k1_INTC_s011 + 1.2781 w_k1_KO_s011
                                 + 1.70362 w_k1_MAT_s011
                                 + 1.14606 w_k1_MSFT_s011 + 1.1976 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.95941 w_k1_ADM_s011
                                 + 0.949657 w_k1_AXP_s011
                                 + 1.08421 w_k1_DIS_s011
                                 + 0.92579 w_k1_FDX_s011
                                 + 1.05454 w_k1_IBM_s011
                                 + 1.34666 w_k1_INTC_s011
                                 + 1.01307 w_k1_KO_s011
                                 + 0.834079 w_k1_MAT_s011
                                 + 1.07891 w_k1_MSFT_s011
                                 + 1.00376 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.07955 w_k1_ADM_s011
                                 + 0.876797 w_k1_AXP_s011
                                 + 1.26049 w_k1_DIS_s011
                                 + 1.06967 w_k1_FDX_s011
                                 + 0.864624 w_k1_IBM_s011
                                 + 1.31645 w_k1_INTC_s011
                                 + 1.14459 w_k1_KO_s011 + 0.7658 w_k1_MAT_s011
                                 + 1.26428 w_k1_MSFT_s011
                                 + 1.16735 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.27507 w_k1_ADM_s011
                                 + 1.4045 w_k1_AXP_s011 + 1.38374 w_k1_DIS_s011
                                 + 1.26695 w_k1_FDX_s011
                                 + 0.977539 w_k1_IBM_s011
                                 + 1.4729 w_k1_INTC_s011 + 1.19264 w_k1_KO_s011
                                 + 1.09554 w_k1_MAT_s011
                                 + 1.14402 w_k1_MSFT_s011
                                 + 1.15925 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.956459 w_k1_ADM_s012
                                 + 0.856508 w_k1_AXP_s012
                                 + 1.01553 w_k1_DIS_s012
                                 + 1.02723 w_k1_FDX_s012
                                 + 1.02417 w_k1_IBM_s012
                                 + 0.958296 w_k1_INTC_s012
                                 + 1.00754 w_k1_KO_s012
                                 + 0.623667 w_k1_MAT_s012
                                 + 1.02961 w_k1_MSFT_s012
                                 + 0.934485 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.04773 w_k1_ADM_s012
                                 + 1.15082 w_k1_AXP_s012
                                 + 1.21653 w_k1_DIS_s012
                                 + 1.14615 w_k1_FDX_s012 + 1.4078 w_k1_IBM_s012
                                 + 1.00633 w_k1_INTC_s012
                                 + 1.17236 w_k1_KO_s012
                                 + 0.770489 w_k1_MAT_s012
                                 + 1.05815 w_k1_MSFT_s012
                                 + 0.990186 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.837737 w_k1_ADM_s012
                                 + 0.746848 w_k1_AXP_s012
                                 + 0.677394 w_k1_DIS_s012
                                 + 0.818842 w_k1_FDX_s012
                                 + 0.859069 w_k1_IBM_s012
                                 + 0.705011 w_k1_INTC_s012
                                 + 0.76149 w_k1_KO_s012
                                 + 0.513104 w_k1_MAT_s012
                                 + 0.820316 w_k1_MSFT_s012
                                 + 0.808702 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.09023 w_k1_ADM_s012
                                 + 1.22915 w_k1_AXP_s012
                                 + 0.838962 w_k1_DIS_s012
                                 + 1.2059 w_k1_FDX_s012 + 0.92128 w_k1_IBM_s012
                                 + 1.17645 w_k1_INTC_s012
                                 + 0.962348 w_k1_KO_s012
                                 + 0.682839 w_k1_MAT_s012
                                 + 1.00866 w_k1_MSFT_s012
                                 + 1.05204 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.00188 w_k1_ADM_s012
                                 + 0.845781 w_k1_AXP_s012
                                 + 1.07861 w_k1_DIS_s012
                                 + 0.919808 w_k1_FDX_s012
                                 + 1.24771 w_k1_IBM_s012
                                 + 1.06765 w_k1_INTC_s012
                                 + 1.22145 w_k1_KO_s012
                                 + 0.630158 w_k1_MAT_s012
                                 + 1.0612 w_k1_MSFT_s012
                                 + 0.967044 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.09905 w_k1_ADM_s012
                                 + 1.14679 w_k1_AXP_s012
                                 + 1.33692 w_k1_DIS_s012 + 1.1882 w_k1_FDX_s012
                                 + 0.980855 w_k1_IBM_s012
                                 + 1.28578 w_k1_INTC_s012
                                 + 1.21113 w_k1_KO_s012 + 1.05832 w_k1_MAT_s012
                                 + 1.23382 w_k1_MSFT_s012
                                 + 1.05948 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.2504 w_k1_ADM_s012
                                 + 1.26224 w_k1_AXP_s012
                                 + 1.80236 w_k1_DIS_s012 + 1.4808 w_k1_FDX_s012
                                 + 1.21338 w_k1_IBM_s012
                                 + 1.30282 w_k1_INTC_s012
                                 + 1.24863 w_k1_KO_s012 + 0.88902 w_k1_MAT_s012
                                 + 1.47158 w_k1_MSFT_s012
                                 + 1.13361 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.0674 w_k1_ADM_s012
                                 + 1.60552 w_k1_AXP_s012
                                 + 1.28725 w_k1_DIS_s012
                                 + 1.25707 w_k1_FDX_s012
                                 + 1.62216 w_k1_IBM_s012
                                 + 1.32644 w_k1_INTC_s012
                                 + 1.34302 w_k1_KO_s012 + 1.85212 w_k1_MAT_s012
                                 + 1.10131 w_k1_MSFT_s012
                                 + 1.09892 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.32923 w_k1_ADM_s013
                                 + 1.26932 w_k1_AXP_s013
                                 + 1.16767 w_k1_DIS_s013
                                 + 1.13158 w_k1_FDX_s013
                                 + 1.03517 w_k1_IBM_s013
                                 + 1.04979 w_k1_INTC_s013
                                 + 1.00944 w_k1_KO_s013
                                 + 0.953725 w_k1_MAT_s013
                                 + 1.06638 w_k1_MSFT_s013
                                 + 1.07963 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.66936 w_k1_ADM_s013
                                 + 1.54247 w_k1_AXP_s013
                                 + 1.46449 w_k1_DIS_s013
                                 + 1.29934 w_k1_FDX_s013
                                 + 1.66945 w_k1_IBM_s013
                                 + 0.983923 w_k1_INTC_s013
                                 + 1.28842 w_k1_KO_s013 + 1.48963 w_k1_MAT_s013
                                 + 1.25986 w_k1_MSFT_s013
                                 + 1.11167 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.835598 w_k1_ADM_s013
                                 + 0.796572 w_k1_AXP_s013
                                 + 0.857858 w_k1_DIS_s013
                                 + 0.776116 w_k1_FDX_s013
                                 + 1.26696 w_k1_IBM_s013
                                 + 0.868125 w_k1_INTC_s013
                                 + 0.980563 w_k1_KO_s013
                                 + 1.21152 w_k1_MAT_s013
                                 + 1.00261 w_k1_MSFT_s013
                                 + 0.992106 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.934909 w_k1_ADM_s013
                                 + 1.11101 w_k1_AXP_s013
                                 + 1.19911 w_k1_DIS_s013
                                 + 0.915864 w_k1_FDX_s013
                                 + 1.60777 w_k1_IBM_s013
                                 + 0.821141 w_k1_INTC_s013
                                 + 1.14295 w_k1_KO_s013 + 1.74001 w_k1_MAT_s013
                                 + 0.97295 w_k1_MSFT_s013
                                 + 1.03293 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.993473 w_k1_ADM_s013
                                 + 0.897397 w_k1_AXP_s013
                                 + 1.02371 w_k1_DIS_s013
                                 + 0.893799 w_k1_FDX_s013
                                 + 0.988253 w_k1_IBM_s013
                                 + 0.985903 w_k1_INTC_s013
                                 + 0.965463 w_k1_KO_s013
                                 + 0.841754 w_k1_MAT_s013
                                 + 0.995133 w_k1_MSFT_s013
                                 + 0.921002 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.854022 w_k1_ADM_s013
                                 + 0.726372 w_k1_AXP_s013
                                 + 0.806195 w_k1_DIS_s013
                                 + 0.865184 w_k1_FDX_s013
                                 + 0.710847 w_k1_IBM_s013
                                 + 0.74797 w_k1_INTC_s013
                                 + 0.752537 w_k1_KO_s013
                                 + 0.613787 w_k1_MAT_s013
                                 + 0.802584 w_k1_MSFT_s013
                                 + 0.840253 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.31655 w_k1_ADM_s013
                                 + 1.24742 w_k1_AXP_s013
                                 + 1.37936 w_k1_DIS_s013
                                 + 1.41296 w_k1_FDX_s013
                                 + 0.740465 w_k1_IBM_s013
                                 + 1.20863 w_k1_INTC_s013
                                 + 0.970222 w_k1_KO_s013
                                 + 0.86436 w_k1_MAT_s013
                                 + 1.27243 w_k1_MSFT_s013
                                 + 1.10817 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.44372 w_k1_ADM_s013
                                 + 1.56753 w_k1_AXP_s013
                                 + 1.70744 w_k1_DIS_s013
                                 + 1.36676 w_k1_FDX_s013
                                 + 1.14026 w_k1_IBM_s013
                                 + 1.22382 w_k1_INTC_s013
                                 + 1.22363 w_k1_KO_s013 + 1.46548 w_k1_MAT_s013
                                 + 1.29307 w_k1_MSFT_s013
                                 + 1.20434 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.717074 w_k1_ADM_s014
                                 + 0.801289 w_k1_AXP_s014
                                 + 0.940932 w_k1_DIS_s014
                                 + 0.871812 w_k1_FDX_s014
                                 + 1.26469 w_k1_IBM_s014
                                 + 0.899628 w_k1_INTC_s014
                                 + 1.11997 w_k1_KO_s014 + 1.20587 w_k1_MAT_s014
                                 + 0.944966 w_k1_MSFT_s014
                                 + 0.856882 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.87174 w_k1_ADM_s014
                                 + 0.948878 w_k1_AXP_s014
                                 + 1.0553 w_k1_DIS_s014
                                 + 0.972168 w_k1_FDX_s014
                                 + 0.892906 w_k1_IBM_s014
                                 + 0.900415 w_k1_INTC_s014
                                 + 1.06622 w_k1_KO_s014
                                 + 0.873782 w_k1_MAT_s014
                                 + 0.976543 w_k1_MSFT_s014
                                 + 0.979189 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.97377 w_k1_ADM_s014
                                 + 1.15864 w_k1_AXP_s014
                                 + 1.11172 w_k1_DIS_s014
                                 + 1.06328 w_k1_FDX_s014
                                 + 1.06248 w_k1_IBM_s014
                                 + 0.943035 w_k1_INTC_s014
                                 + 1.09995 w_k1_KO_s014 + 1.09075 w_k1_MAT_s014
                                 + 1.07355 w_k1_MSFT_s014
                                 + 0.941737 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.05977 w_k1_ADM_s014
                                 + 0.905144 w_k1_AXP_s014
                                 + 0.982535 w_k1_DIS_s014
                                 + 1.10573 w_k1_FDX_s014
                                 + 0.76323 w_k1_IBM_s014
                                 + 0.988772 w_k1_INTC_s014
                                 + 0.936691 w_k1_KO_s014
                                 + 0.785336 w_k1_MAT_s014
                                 + 0.966055 w_k1_MSFT_s014
                                 + 0.996774 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.916358 w_k1_ADM_s014
                                 + 1.17241 w_k1_AXP_s014
                                 + 1.52595 w_k1_DIS_s014
                                 + 1.30407 w_k1_FDX_s014
                                 + 1.18458 w_k1_IBM_s014
                                 + 1.19523 w_k1_INTC_s014
                                 + 1.38833 w_k1_KO_s014 + 1.30149 w_k1_MAT_s014
                                 + 1.25886 w_k1_MSFT_s014
                                 + 1.10298 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.675123 w_k1_ADM_s014
                                 + 0.66538 w_k1_AXP_s014
                                 + 0.580231 w_k1_DIS_s014
                                 + 0.644733 w_k1_FDX_s014
                                 + 0.980934 w_k1_IBM_s014
                                 + 0.792809 w_k1_INTC_s014
                                 + 0.834208 w_k1_KO_s014
                                 + 0.878718 w_k1_MAT_s014
                                 + 0.769811 w_k1_MSFT_s014
                                 + 0.850171 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.735481 w_k1_ADM_s014
                                 + 0.843916 w_k1_AXP_s014
                                 + 0.681744 w_k1_DIS_s014
                                 + 0.785874 w_k1_FDX_s014
                                 + 0.892527 w_k1_IBM_s014
                                 + 0.781305 w_k1_INTC_s014
                                 + 0.851451 w_k1_KO_s014
                                 + 0.924173 w_k1_MAT_s014
                                 + 0.82206 w_k1_MSFT_s014
                                 + 0.895846 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.06766 w_k1_ADM_s014
                                 + 1.12753 w_k1_AXP_s014
                                 + 0.952602 w_k1_DIS_s014
                                 + 1.04146 w_k1_FDX_s014
                                 + 1.45563 w_k1_IBM_s014
                                 + 0.939748 w_k1_INTC_s014
                                 + 1.30185 w_k1_KO_s014 + 2.0205 w_k1_MAT_s014
                                 + 1.08821 w_k1_MSFT_s014
                                 + 1.11001 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.846879 w_k1_ADM_s015
                                 + 0.740227 w_k1_AXP_s015
                                 + 1.09063 w_k1_DIS_s015
                                 + 1.02924 w_k1_FDX_s015
                                 + 0.631587 w_k1_IBM_s015
                                 + 1.04782 w_k1_INTC_s015
                                 + 0.899637 w_k1_KO_s015
                                 + 0.874398 w_k1_MAT_s015
                                 + 1.18785 w_k1_MSFT_s015
                                 + 0.987961 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.03896 w_k1_ADM_s015
                                 + 1.22364 w_k1_AXP_s015
                                 + 1.89397 w_k1_DIS_s015 + 1.3545 w_k1_FDX_s015
                                 + 1.03446 w_k1_IBM_s015
                                 + 1.47183 w_k1_INTC_s015
                                 + 1.23536 w_k1_KO_s015 + 1.01142 w_k1_MAT_s015
                                 + 1.51904 w_k1_MSFT_s015
                                 + 1.17192 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.816788 w_k1_ADM_s015
                                 + 0.811092 w_k1_AXP_s015
                                 + 0.67086 w_k1_DIS_s015
                                 + 0.888946 w_k1_FDX_s015
                                 + 0.683722 w_k1_IBM_s015
                                 + 0.960373 w_k1_INTC_s015
                                 + 0.734986 w_k1_KO_s015
                                 + 0.851165 w_k1_MAT_s015
                                 + 0.844635 w_k1_MSFT_s015
                                 + 0.894282 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.05704 w_k1_ADM_s015
                                 + 0.979387 w_k1_AXP_s015
                                 + 1.1645 w_k1_DIS_s015 + 1.08808 w_k1_FDX_s015
                                 + 1.55229 w_k1_IBM_s015
                                 + 1.04633 w_k1_INTC_s015
                                 + 1.31704 w_k1_KO_s015 + 1.76607 w_k1_MAT_s015
                                 + 1.20872 w_k1_MSFT_s015
                                 + 1.20378 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.742479 w_k1_ADM_s015
                                 + 0.885604 w_k1_AXP_s015
                                 + 0.945747 w_k1_DIS_s015
                                 + 0.918001 w_k1_FDX_s015
                                 + 0.878548 w_k1_IBM_s015
                                 + 0.910371 w_k1_INTC_s015
                                 + 0.85105 w_k1_KO_s015 + 1.06374 w_k1_MAT_s015
                                 + 0.978832 w_k1_MSFT_s015
                                 + 0.990557 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.963876 w_k1_ADM_s015
                                 + 0.935542 w_k1_AXP_s015
                                 + 1.19137 w_k1_DIS_s015 + 1.0663 w_k1_FDX_s015
                                 + 0.968903 w_k1_IBM_s015
                                 + 1.13848 w_k1_INTC_s015
                                 + 0.913536 w_k1_KO_s015
                                 + 0.769939 w_k1_MAT_s015
                                 + 1.05532 w_k1_MSFT_s015
                                 + 1.02409 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.00524 w_k1_ADM_s015
                                 + 1.32368 w_k1_AXP_s015 + 1.3302 w_k1_DIS_s015
                                 + 1.26974 w_k1_FDX_s015
                                 + 1.12217 w_k1_IBM_s015
                                 + 1.0832 w_k1_INTC_s015 + 1.04857 w_k1_KO_s015
                                 + 1.63598 w_k1_MAT_s015
                                 + 1.11232 w_k1_MSFT_s015
                                 + 1.26131 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.740487 w_k1_ADM_s015
                                 + 0.599164 w_k1_AXP_s015
                                 + 0.709511 w_k1_DIS_s015
                                 + 0.751584 w_k1_FDX_s015
                                 + 1.13882 w_k1_IBM_s015
                                 + 0.952525 w_k1_INTC_s015
                                 + 0.871913 w_k1_KO_s015
                                 + 0.907258 w_k1_MAT_s015
                                 + 0.873275 w_k1_MSFT_s015
                                 + 0.910375 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.0161 w_k1_ADM_s016
                                 + 1.2214 w_k1_AXP_s016
                                 + 0.929686 w_k1_DIS_s016
                                 + 1.04477 w_k1_FDX_s016
                                 + 1.28603 w_k1_IBM_s016
                                 + 0.824446 w_k1_INTC_s016
                                 + 1.04849 w_k1_KO_s016 + 1.16039 w_k1_MAT_s016
                                 + 1.00354 w_k1_MSFT_s016
                                 + 1.09186 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05597 w_k1_ADM_s016
                                 + 1.10786 w_k1_AXP_s016
                                 + 1.34058 w_k1_DIS_s016 + 1.2399 w_k1_FDX_s016
                                 + 1.54485 w_k1_IBM_s016
                                 + 1.03471 w_k1_INTC_s016
                                 + 1.31129 w_k1_KO_s016 + 1.70312 w_k1_MAT_s016
                                 + 1.28668 w_k1_MSFT_s016
                                 + 1.31098 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.719849 w_k1_ADM_s016
                                 + 0.810298 w_k1_AXP_s016
                                 + 0.820664 w_k1_DIS_s016
                                 + 0.906672 w_k1_FDX_s016
                                 + 0.89421 w_k1_IBM_s016
                                 + 0.878641 w_k1_INTC_s016
                                 + 0.952825 w_k1_KO_s016
                                 + 0.999096 w_k1_MAT_s016
                                 + 0.979495 w_k1_MSFT_s016
                                 + 0.882143 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.614226 w_k1_ADM_s016
                                 + 0.59454 w_k1_AXP_s016
                                 + 0.586423 w_k1_DIS_s016
                                 + 0.737175 w_k1_FDX_s016
                                 + 0.762568 w_k1_IBM_s016
                                 + 0.693262 w_k1_INTC_s016
                                 + 0.656192 w_k1_KO_s016
                                 + 0.858877 w_k1_MAT_s016
                                 + 0.831086 w_k1_MSFT_s016
                                 + 0.803016 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.20857 w_k1_ADM_s016
                                 + 1.60499 w_k1_AXP_s016
                                 + 1.43034 w_k1_DIS_s016
                                 + 1.37048 w_k1_FDX_s016
                                 + 1.12828 w_k1_IBM_s016
                                 + 1.01036 w_k1_INTC_s016
                                 + 1.06054 w_k1_KO_s016 + 1.51482 w_k1_MAT_s016
                                 + 1.2485 w_k1_MSFT_s016 + 1.12706 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.04488 w_k1_ADM_s016
                                 + 0.996065 w_k1_AXP_s016
                                 + 0.954233 w_k1_DIS_s016
                                 + 1.03898 w_k1_FDX_s016
                                 + 0.702295 w_k1_IBM_s016
                                 + 0.702723 w_k1_INTC_s016
                                 + 0.69587 w_k1_KO_s016
                                 + 0.640377 w_k1_MAT_s016
                                 + 0.86364 w_k1_MSFT_s016
                                 + 0.860351 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.11725 w_k1_ADM_s016
                                 + 1.07637 w_k1_AXP_s016
                                 + 0.974664 w_k1_DIS_s016
                                 + 1.09045 w_k1_FDX_s016
                                 + 1.02654 w_k1_IBM_s016
                                 + 0.95959 w_k1_INTC_s016
                                 + 0.952781 w_k1_KO_s016
                                 + 0.985161 w_k1_MAT_s016
                                 + 1.02192 w_k1_MSFT_s016
                                 + 1.06598 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.561649 w_k1_ADM_s016
                                 + 0.696751 w_k1_AXP_s016
                                 + 0.628837 w_k1_DIS_s016
                                 + 0.71513 w_k1_FDX_s016
                                 + 1.15685 w_k1_IBM_s016
                                 + 0.620636 w_k1_INTC_s016
                                 + 0.887933 w_k1_KO_s016
                                 + 1.31542 w_k1_MAT_s016
                                 + 0.755833 w_k1_MSFT_s016
                                 + 0.884089 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.975853 w_k1_ADM_s017
                                 + 0.92039 w_k1_AXP_s017
                                 + 0.530547 w_k1_DIS_s017
                                 + 0.831551 w_k1_FDX_s017
                                 + 0.83066 w_k1_IBM_s017
                                 + 0.772185 w_k1_INTC_s017
                                 + 0.643727 w_k1_KO_s017
                                 + 0.60806 w_k1_MAT_s017
                                 + 0.641155 w_k1_MSFT_s017
                                 + 0.832726 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.819272 w_k1_ADM_s017
                                 + 0.748454 w_k1_AXP_s017
                                 + 0.695874 w_k1_DIS_s017
                                 + 0.816147 w_k1_FDX_s017
                                 + 0.895187 w_k1_IBM_s017
                                 + 0.869176 w_k1_INTC_s017
                                 + 0.873403 w_k1_KO_s017
                                 + 0.667793 w_k1_MAT_s017
                                 + 0.875392 w_k1_MSFT_s017
                                 + 0.984419 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.03384 w_k1_ADM_s017
                                 + 0.884353 w_k1_AXP_s017
                                 + 1.04956 w_k1_DIS_s017
                                 + 1.09494 w_k1_FDX_s017
                                 + 0.804167 w_k1_IBM_s017
                                 + 1.02406 w_k1_INTC_s017
                                 + 0.988051 w_k1_KO_s017
                                 + 0.529667 w_k1_MAT_s017
                                 + 1.01638 w_k1_MSFT_s017
                                 + 0.989464 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00844 w_k1_ADM_s017
                                 + 1.03097 w_k1_AXP_s017
                                 + 0.709653 w_k1_DIS_s017
                                 + 0.992965 w_k1_FDX_s017
                                 + 0.653878 w_k1_IBM_s017
                                 + 0.959441 w_k1_INTC_s017
                                 + 0.667336 w_k1_KO_s017
                                 + 0.662433 w_k1_MAT_s017
                                 + 0.856274 w_k1_MSFT_s017
                                 + 0.921386 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.27743 w_k1_ADM_s017
                                 + 1.09127 w_k1_AXP_s017
                                 + 1.44348 w_k1_DIS_s017
                                 + 1.17492 w_k1_FDX_s017
                                 + 1.13103 w_k1_IBM_s017
                                 + 1.16498 w_k1_INTC_s017
                                 + 1.18622 w_k1_KO_s017 + 1.01653 w_k1_MAT_s017
                                 + 1.12694 w_k1_MSFT_s017 + 1.1537 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.20251 w_k1_ADM_s017
                                 + 1.18384 w_k1_AXP_s017
                                 + 1.15998 w_k1_DIS_s017
                                 + 1.00489 w_k1_FDX_s017
                                 + 1.84083 w_k1_IBM_s017
                                 + 1.16165 w_k1_INTC_s017
                                 + 1.22844 w_k1_KO_s017 + 1.19525 w_k1_MAT_s017
                                 + 1.13278 w_k1_MSFT_s017
                                 + 1.05534 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.961506 w_k1_ADM_s017
                                 + 1.19319 w_k1_AXP_s017
                                 + 0.966724 w_k1_DIS_s017
                                 + 0.984285 w_k1_FDX_s017
                                 + 1.2851 w_k1_IBM_s017
                                 + 1.10906 w_k1_INTC_s017
                                 + 1.02636 w_k1_KO_s017 + 1.21528 w_k1_MAT_s017
                                 + 1.0094 w_k1_MSFT_s017 + 1.05473 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.30688 w_k1_ADM_s017
                                 + 1.59541 w_k1_AXP_s017
                                 + 1.71576 w_k1_DIS_s017
                                 + 1.47214 w_k1_FDX_s017
                                 + 1.26422 w_k1_IBM_s017
                                 + 1.38457 w_k1_INTC_s017 + 1.3182 w_k1_KO_s017
                                 + 1.26558 w_k1_MAT_s017
                                 + 1.24102 w_k1_MSFT_s017
                                 + 1.24449 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.919576 w_k1_ADM_s100
                                 + 0.759868 w_k1_AXP_s100
                                 + 0.70659 w_k1_DIS_s100
                                 + 0.794901 w_k1_FDX_s100
                                 + 0.709885 w_k1_IBM_s100
                                 + 1.11127 w_k1_INTC_s100
                                 + 0.71162 w_k1_KO_s100
                                 + 0.477346 w_k1_MAT_s100
                                 + 0.764838 w_k1_MSFT_s100
                                 + 0.751183 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.79722 w_k1_ADM_s100
                                 + 0.858379 w_k1_AXP_s100
                                 + 0.745398 w_k1_DIS_s100
                                 + 0.786824 w_k1_FDX_s100
                                 + 1.12822 w_k1_IBM_s100
                                 + 0.880369 w_k1_INTC_s100
                                 + 0.820918 w_k1_KO_s100
                                 + 1.15824 w_k1_MAT_s100
                                 + 0.847822 w_k1_MSFT_s100
                                 + 0.838653 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.975489 w_k1_ADM_s100
                                 + 1.02022 w_k1_AXP_s100
                                 + 0.967331 w_k1_DIS_s100
                                 + 0.923343 w_k1_FDX_s100
                                 + 1.16478 w_k1_IBM_s100
                                 + 1.15232 w_k1_INTC_s100
                                 + 1.04764 w_k1_KO_s100
                                 + 0.720512 w_k1_MAT_s100
                                 + 0.923024 w_k1_MSFT_s100
                                 + 1.01574 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.23994 w_k1_ADM_s100
                                 + 1.16295 w_k1_AXP_s100
                                 + 1.12094 w_k1_DIS_s100
                                 + 1.27032 w_k1_FDX_s100
                                 + 0.735024 w_k1_IBM_s100
                                 + 1.1612 w_k1_INTC_s100
                                 + 0.911062 w_k1_KO_s100
                                 + 0.793036 w_k1_MAT_s100
                                 + 1.2534 w_k1_MSFT_s100
                                 + 0.956752 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.752298 w_k1_ADM_s100
                                 + 0.73969 w_k1_AXP_s100
                                 + 0.815279 w_k1_DIS_s100
                                 + 0.820878 w_k1_FDX_s100
                                 + 1.31904 w_k1_IBM_s100
                                 + 0.891349 w_k1_INTC_s100
                                 + 1.1294 w_k1_KO_s100 + 1.41259 w_k1_MAT_s100
                                 + 1.00749 w_k1_MSFT_s100
                                 + 0.909056 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.00181 w_k1_ADM_s100
                                 + 1.40544 w_k1_AXP_s100 + 1.2045 w_k1_DIS_s100
                                 + 1.08657 w_k1_FDX_s100
                                 + 1.59164 w_k1_IBM_s100
                                 + 1.23917 w_k1_INTC_s100
                                 + 1.37067 w_k1_KO_s100 + 1.9931 w_k1_MAT_s100
                                 + 1.10624 w_k1_MSFT_s100
                                 + 1.14984 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.05006 w_k1_ADM_s100
                                 + 1.10607 w_k1_AXP_s100
                                 + 1.15755 w_k1_DIS_s100
                                 + 1.07938 w_k1_FDX_s100
                                 + 0.823652 w_k1_IBM_s100
                                 + 1.28996 w_k1_INTC_s100
                                 + 0.87998 w_k1_KO_s100
                                 + 0.764736 w_k1_MAT_s100
                                 + 1.25559 w_k1_MSFT_s100
                                 + 0.987871 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.40327 w_k1_ADM_s100
                                 + 1.22607 w_k1_AXP_s100
                                 + 1.58225 w_k1_DIS_s100
                                 + 1.17327 w_k1_FDX_s100
                                 + 1.20474 w_k1_IBM_s100
                                 + 1.26825 w_k1_INTC_s100
                                 + 1.31566 w_k1_KO_s100 + 1.37214 w_k1_MAT_s100
                                 + 1.34333 w_k1_MSFT_s100
                                 + 1.13012 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.768538 w_k1_ADM_s101
                                 + 0.841845 w_k1_AXP_s101
                                 + 0.717661 w_k1_DIS_s101
                                 + 0.808352 w_k1_FDX_s101
                                 + 0.788436 w_k1_IBM_s101
                                 + 0.941649 w_k1_INTC_s101
                                 + 0.697627 w_k1_KO_s101
                                 + 0.914981 w_k1_MAT_s101
                                 + 0.695059 w_k1_MSFT_s101
                                 + 0.905133 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.912501 w_k1_ADM_s101
                                 + 1.85432 w_k1_AXP_s101
                                 + 1.26496 w_k1_DIS_s101 + 1.2381 w_k1_FDX_s101
                                 + 1.61424 w_k1_IBM_s101
                                 + 1.27383 w_k1_INTC_s101
                                 + 1.19037 w_k1_KO_s101 + 2.33277 w_k1_MAT_s101
                                 + 1.08548 w_k1_MSFT_s101 + 1.1817 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.999222 w_k1_ADM_s101
                                 + 0.951553 w_k1_AXP_s101
                                 + 1.19422 w_k1_DIS_s101
                                 + 1.07303 w_k1_FDX_s101
                                 + 0.769485 w_k1_IBM_s101
                                 + 1.17492 w_k1_INTC_s101
                                 + 0.922179 w_k1_KO_s101
                                 + 0.958743 w_k1_MAT_s101
                                 + 1.08068 w_k1_MSFT_s101
                                 + 0.922218 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.94481 w_k1_ADM_s101
                                 + 1.07493 w_k1_AXP_s101
                                 + 1.27741 w_k1_DIS_s101
                                 + 0.934936 w_k1_FDX_s101
                                 + 1.7672 w_k1_IBM_s101
                                 + 1.10934 w_k1_INTC_s101
                                 + 1.46976 w_k1_KO_s101 + 1.61822 w_k1_MAT_s101
                                 + 1.16024 w_k1_MSFT_s101
                                 + 1.09406 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.04514 w_k1_ADM_s101
                                 + 1.2615 w_k1_AXP_s101 + 1.3668 w_k1_DIS_s101
                                 + 1.23662 w_k1_FDX_s101
                                 + 1.22002 w_k1_IBM_s101
                                 + 1.2155 w_k1_INTC_s101 + 1.27832 w_k1_KO_s101
                                 + 1.60121 w_k1_MAT_s101
                                 + 1.1406 w_k1_MSFT_s101 + 1.11052 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.727877 w_k1_ADM_s101
                                 + 0.758326 w_k1_AXP_s101
                                 + 0.503812 w_k1_DIS_s101
                                 + 0.648873 w_k1_FDX_s101
                                 + 0.717042 w_k1_IBM_s101
                                 + 0.70705 w_k1_INTC_s101
                                 + 0.56953 w_k1_KO_s101
                                 + 0.685748 w_k1_MAT_s101
                                 + 0.588978 w_k1_MSFT_s101
                                 + 0.697508 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.955357 w_k1_ADM_s101
                                 + 0.934189 w_k1_AXP_s101
                                 + 0.986632 w_k1_DIS_s101
                                 + 0.841296 w_k1_FDX_s101
                                 + 1.28611 w_k1_IBM_s101
                                 + 1.04937 w_k1_INTC_s101
                                 + 1.06137 w_k1_KO_s101 + 1.4022 w_k1_MAT_s101
                                 + 0.856553 w_k1_MSFT_s101
                                 + 1.00389 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.953998 w_k1_ADM_s101
                                 + 1.02633 w_k1_AXP_s101
                                 + 0.863958 w_k1_DIS_s101
                                 + 0.915478 w_k1_FDX_s101
                                 + 1.09225 w_k1_IBM_s101
                                 + 0.934514 w_k1_INTC_s101
                                 + 0.942404 w_k1_KO_s101
                                 + 1.58861 w_k1_MAT_s101
                                 + 0.824356 w_k1_MSFT_s101
                                 + 0.977888 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.760102 w_k1_ADM_s102
                                 + 0.630418 w_k1_AXP_s102
                                 + 0.620914 w_k1_DIS_s102
                                 + 0.678143 w_k1_FDX_s102
                                 + 0.758235 w_k1_IBM_s102
                                 + 0.795904 w_k1_INTC_s102
                                 + 0.808617 w_k1_KO_s102
                                 + 0.886587 w_k1_MAT_s102
                                 + 0.799974 w_k1_MSFT_s102
                                 + 0.817908 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.15268 w_k1_ADM_s102
                                 + 1.12684 w_k1_AXP_s102
                                 + 1.10959 w_k1_DIS_s102
                                 + 1.17191 w_k1_FDX_s102
                                 + 0.902737 w_k1_IBM_s102
                                 + 1.36111 w_k1_INTC_s102
                                 + 1.21404 w_k1_KO_s102
                                 + 0.751017 w_k1_MAT_s102
                                 + 1.15511 w_k1_MSFT_s102
                                 + 1.09574 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02231 w_k1_ADM_s102
                                 + 0.872933 w_k1_AXP_s102
                                 + 1.16809 w_k1_DIS_s102
                                 + 0.995902 w_k1_FDX_s102
                                 + 1.38385 w_k1_IBM_s102
                                 + 0.966332 w_k1_INTC_s102
                                 + 1.30411 w_k1_KO_s102 + 1.37495 w_k1_MAT_s102
                                 + 1.0268 w_k1_MSFT_s102 + 1.13301 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.938698 w_k1_ADM_s102
                                 + 1.00368 w_k1_AXP_s102
                                 + 0.921279 w_k1_DIS_s102
                                 + 0.966077 w_k1_FDX_s102
                                 + 0.815005 w_k1_IBM_s102
                                 + 1.10376 w_k1_INTC_s102 + 1.0339 w_k1_KO_s102
                                 + 0.808004 w_k1_MAT_s102
                                 + 0.94033 w_k1_MSFT_s102
                                 + 0.875355 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.09583 w_k1_ADM_s102
                                 + 1.27213 w_k1_AXP_s102
                                 + 1.54861 w_k1_DIS_s102
                                 + 1.38756 w_k1_FDX_s102
                                 + 1.12347 w_k1_IBM_s102
                                 + 1.28686 w_k1_INTC_s102
                                 + 1.26807 w_k1_KO_s102 + 1.24047 w_k1_MAT_s102
                                 + 1.42231 w_k1_MSFT_s102 + 1.1026 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.75647 w_k1_ADM_s102
                                 + 1.54037 w_k1_AXP_s102
                                 + 1.80596 w_k1_DIS_s102
                                 + 1.38848 w_k1_FDX_s102
                                 + 1.43928 w_k1_IBM_s102
                                 + 1.05035 w_k1_INTC_s102
                                 + 1.57821 w_k1_KO_s102 + 1.63562 w_k1_MAT_s102
                                 + 1.33019 w_k1_MSFT_s102 + 1.3214 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.01113 w_k1_ADM_s102
                                 + 0.841869 w_k1_AXP_s102
                                 + 0.976486 w_k1_DIS_s102
                                 + 0.956963 w_k1_FDX_s102
                                 + 0.864005 w_k1_IBM_s102
                                 + 1.08101 w_k1_INTC_s102
                                 + 1.07464 w_k1_KO_s102
                                 + 0.607351 w_k1_MAT_s102
                                 + 1.0531 w_k1_MSFT_s102 + 1.03133 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.952908 w_k1_ADM_s102
                                 + 0.761767 w_k1_AXP_s102
                                 + 0.655427 w_k1_DIS_s102
                                 + 0.759704 w_k1_FDX_s102
                                 + 1.11532 w_k1_IBM_s102
                                 + 0.883928 w_k1_INTC_s102
                                 + 0.968755 w_k1_KO_s102
                                 + 1.10945 w_k1_MAT_s102
                                 + 0.872896 w_k1_MSFT_s102
                                 + 0.904515 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.830266 w_k1_ADM_s103
                                 + 0.809406 w_k1_AXP_s103
                                 + 0.762431 w_k1_DIS_s103
                                 + 0.782857 w_k1_FDX_s103
                                 + 1.00317 w_k1_IBM_s103
                                 + 0.911068 w_k1_INTC_s103
                                 + 0.919039 w_k1_KO_s103
                                 + 1.24918 w_k1_MAT_s103
                                 + 0.919096 w_k1_MSFT_s103
                                 + 0.886919 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.19901 w_k1_ADM_s103
                                 + 1.61568 w_k1_AXP_s103
                                 + 1.63243 w_k1_DIS_s103 + 1.4156 w_k1_FDX_s103
                                 + 1.21944 w_k1_IBM_s103
                                 + 1.3419 w_k1_INTC_s103 + 1.37054 w_k1_KO_s103
                                 + 1.3256 w_k1_MAT_s103
                                 + 1.51286 w_k1_MSFT_s103
                                 + 1.28514 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.936873 w_k1_ADM_s103
                                 + 0.986004 w_k1_AXP_s103
                                 + 1.18183 w_k1_DIS_s103
                                 + 1.01528 w_k1_FDX_s103
                                 + 1.29767 w_k1_IBM_s103
                                 + 0.889197 w_k1_INTC_s103
                                 + 1.20523 w_k1_KO_s103 + 1.46992 w_k1_MAT_s103
                                 + 1.09122 w_k1_MSFT_s103
                                 + 1.14848 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.997859 w_k1_ADM_s103
                                 + 1.46391 w_k1_AXP_s103
                                 + 1.38581 w_k1_DIS_s103
                                 + 1.17584 w_k1_FDX_s103
                                 + 1.73234 w_k1_IBM_s103
                                 + 1.11895 w_k1_INTC_s103
                                 + 1.32457 w_k1_KO_s103 + 2.11434 w_k1_MAT_s103
                                 + 1.24884 w_k1_MSFT_s103
                                 + 1.18236 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.972231 w_k1_ADM_s103
                                 + 0.753242 w_k1_AXP_s103
                                 + 0.737584 w_k1_DIS_s103
                                 + 0.807411 w_k1_FDX_s103
                                 + 0.729249 w_k1_IBM_s103
                                 + 0.769883 w_k1_INTC_s103
                                 + 0.799514 w_k1_KO_s103
                                 + 0.848928 w_k1_MAT_s103
                                 + 0.826803 w_k1_MSFT_s103
                                 + 0.799344 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.04514 w_k1_ADM_s103
                                 + 1.20342 w_k1_AXP_s103
                                 + 0.919833 w_k1_DIS_s103
                                 + 1.07863 w_k1_FDX_s103
                                 + 0.877721 w_k1_IBM_s103
                                 + 0.976824 w_k1_INTC_s103
                                 + 0.774378 w_k1_KO_s103
                                 + 1.22628 w_k1_MAT_s103
                                 + 0.962572 w_k1_MSFT_s103
                                 + 0.914649 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.781648 w_k1_ADM_s103
                                 + 0.60325 w_k1_AXP_s103
                                 + 0.607225 w_k1_DIS_s103
                                 + 0.710387 w_k1_FDX_s103
                                 + 0.854426 w_k1_IBM_s103
                                 + 0.71001 w_k1_INTC_s103
                                 + 0.693954 w_k1_KO_s103
                                 + 0.647202 w_k1_MAT_s103
                                 + 0.704349 w_k1_MSFT_s103
                                 + 0.780655 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.990908 w_k1_ADM_s103
                                 + 1.18056 w_k1_AXP_s103
                                 + 1.17646 w_k1_DIS_s103
                                 + 1.08487 w_k1_FDX_s103
                                 + 1.15571 w_k1_IBM_s103
                                 + 1.01199 w_k1_INTC_s103 + 1.177 w_k1_KO_s103
                                 + 1.59402 w_k1_MAT_s103
                                 + 1.09213 w_k1_MSFT_s103
                                 + 1.07172 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.768094 w_k1_ADM_s104
                                 + 0.689581 w_k1_AXP_s104
                                 + 0.697499 w_k1_DIS_s104
                                 + 0.702866 w_k1_FDX_s104
                                 + 0.911927 w_k1_IBM_s104
                                 + 0.805599 w_k1_INTC_s104
                                 + 0.938277 w_k1_KO_s104 + 1.0374 w_k1_MAT_s104
                                 + 0.783167 w_k1_MSFT_s104
                                 + 0.837508 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.968837 w_k1_ADM_s104
                                 + 0.970103 w_k1_AXP_s104
                                 + 1.10502 w_k1_DIS_s104
                                 + 0.854067 w_k1_FDX_s104
                                 + 1.47638 w_k1_IBM_s104
                                 + 0.920568 w_k1_INTC_s104
                                 + 1.27733 w_k1_KO_s104 + 2.35231 w_k1_MAT_s104
                                 + 0.957939 w_k1_MSFT_s104
                                 + 1.02194 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.929309 w_k1_ADM_s104
                                 + 0.778336 w_k1_AXP_s104
                                 + 0.838519 w_k1_DIS_s104
                                 + 1.00581 w_k1_FDX_s104
                                 + 0.614196 w_k1_IBM_s104
                                 + 1.08771 w_k1_INTC_s104
                                 + 0.848596 w_k1_KO_s104
                                 + 0.769049 w_k1_MAT_s104
                                 + 1.01868 w_k1_MSFT_s104
                                 + 0.962906 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.988157 w_k1_ADM_s104
                                 + 0.756622 w_k1_AXP_s104
                                 + 0.781453 w_k1_DIS_s104
                                 + 0.897378 w_k1_FDX_s104
                                 + 0.706012 w_k1_IBM_s104
                                 + 0.87018 w_k1_INTC_s104
                                 + 0.90255 w_k1_KO_s104
                                 + 0.754036 w_k1_MAT_s104
                                 + 0.882631 w_k1_MSFT_s104
                                 + 0.93722 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.24685 w_k1_ADM_s104
                                 + 0.953729 w_k1_AXP_s104
                                 + 1.19689 w_k1_DIS_s104 + 1.0478 w_k1_FDX_s104
                                 + 1.03405 w_k1_IBM_s104
                                 + 0.863471 w_k1_INTC_s104
                                 + 1.15235 w_k1_KO_s104 + 1.10877 w_k1_MAT_s104
                                 + 1.0486 w_k1_MSFT_s104 + 1.11251 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.31536 w_k1_ADM_s104
                                 + 0.887659 w_k1_AXP_s104
                                 + 1.15903 w_k1_DIS_s104
                                 + 1.22834 w_k1_FDX_s104
                                 + 0.70205 w_k1_IBM_s104
                                 + 1.0918 w_k1_INTC_s104 + 1.04687 w_k1_KO_s104
                                 + 0.717245 w_k1_MAT_s104
                                 + 1.22849 w_k1_MSFT_s104
                                 + 1.03496 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.27228 w_k1_ADM_s104
                                 + 1.48874 w_k1_AXP_s104
                                 + 1.31662 w_k1_DIS_s104
                                 + 1.10125 w_k1_FDX_s104
                                 + 1.41263 w_k1_IBM_s104
                                 + 1.27528 w_k1_INTC_s104
                                 + 1.27624 w_k1_KO_s104 + 1.6632 w_k1_MAT_s104
                                 + 1.09092 w_k1_MSFT_s104 + 1.1524 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.5771 w_k1_ADM_s104
                                 + 1.3514 w_k1_AXP_s104 + 1.62294 w_k1_DIS_s104
                                 + 1.38748 w_k1_FDX_s104
                                 + 0.989015 w_k1_IBM_s104
                                 + 1.1229 w_k1_INTC_s104 + 1.35682 w_k1_KO_s104
                                 + 1.4309 w_k1_MAT_s104
                                 + 1.31035 w_k1_MSFT_s104
                                 + 1.16702 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.2448 w_k1_ADM_s105
                                 + 1.27025 w_k1_AXP_s105
                                 + 1.67886 w_k1_DIS_s105
                                 + 1.27815 w_k1_FDX_s105
                                 + 1.09611 w_k1_IBM_s105
                                 + 1.12997 w_k1_INTC_s105
                                 + 1.20275 w_k1_KO_s105
                                 + 0.798448 w_k1_MAT_s105
                                 + 1.22469 w_k1_MSFT_s105
                                 + 1.17298 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.10695 w_k1_ADM_s105
                                 + 0.998075 w_k1_AXP_s105
                                 + 0.935748 w_k1_DIS_s105
                                 + 1.01277 w_k1_FDX_s105
                                 + 0.689466 w_k1_IBM_s105
                                 + 0.967601 w_k1_INTC_s105
                                 + 0.849822 w_k1_KO_s105
                                 + 0.534963 w_k1_MAT_s105
                                 + 1.04547 w_k1_MSFT_s105 + 1.0512 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.07222 w_k1_ADM_s105
                                 + 1.42528 w_k1_AXP_s105
                                 + 1.25566 w_k1_DIS_s105
                                 + 1.12217 w_k1_FDX_s105
                                 + 1.67974 w_k1_IBM_s105
                                 + 0.977918 w_k1_INTC_s105
                                 + 1.40745 w_k1_KO_s105 + 1.78101 w_k1_MAT_s105
                                 + 1.0929 w_k1_MSFT_s105 + 1.12543 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.786853 w_k1_ADM_s105
                                 + 0.73596 w_k1_AXP_s105
                                 + 0.675231 w_k1_DIS_s105
                                 + 0.695616 w_k1_FDX_s105
                                 + 0.74252 w_k1_IBM_s105
                                 + 0.781385 w_k1_INTC_s105
                                 + 0.70041 w_k1_KO_s105 + 0.52384 w_k1_MAT_s105
                                 + 0.748188 w_k1_MSFT_s105
                                 + 0.739091 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.915246 w_k1_ADM_s105
                                 + 1.01728 w_k1_AXP_s105
                                 + 0.825089 w_k1_DIS_s105
                                 + 0.924776 w_k1_FDX_s105
                                 + 0.932282 w_k1_IBM_s105
                                 + 0.901325 w_k1_INTC_s105
                                 + 0.813265 w_k1_KO_s105
                                 + 0.942586 w_k1_MAT_s105
                                 + 0.892309 w_k1_MSFT_s105
                                 + 0.880098 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.983976 w_k1_ADM_s105
                                 + 0.691679 w_k1_AXP_s105
                                 + 0.891359 w_k1_DIS_s105
                                 + 0.922952 w_k1_FDX_s105
                                 + 0.61291 w_k1_IBM_s105
                                 + 0.738982 w_k1_INTC_s105
                                 + 0.782024 w_k1_KO_s105
                                 + 0.474818 w_k1_MAT_s105
                                 + 0.996635 w_k1_MSFT_s105
                                 + 0.908531 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.23448 w_k1_ADM_s105
                                 + 1.4606 w_k1_AXP_s105 + 1.39655 w_k1_DIS_s105
                                 + 1.34724 w_k1_FDX_s105
                                 + 1.00049 w_k1_IBM_s105
                                 + 1.10654 w_k1_INTC_s105
                                 + 1.08028 w_k1_KO_s105 + 1.18876 w_k1_MAT_s105
                                 + 1.2006 w_k1_MSFT_s105 + 1.08108 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.24591 w_k1_ADM_s105
                                 + 1.00844 w_k1_AXP_s105
                                 + 1.01229 w_k1_DIS_s105
                                 + 0.99795 w_k1_FDX_s105
                                 + 1.22557 w_k1_IBM_s105
                                 + 0.8973 w_k1_INTC_s105 + 1.1156 w_k1_KO_s105
                                 + 1.05477 w_k1_MAT_s105
                                 + 1.03963 w_k1_MSFT_s105
                                 + 0.933296 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.07778 w_k1_ADM_s106
                                 + 1.29917 w_k1_AXP_s106
                                 + 1.81876 w_k1_DIS_s106
                                 + 1.41379 w_k1_FDX_s106
                                 + 1.28078 w_k1_IBM_s106
                                 + 1.19091 w_k1_INTC_s106 + 1.5148 w_k1_KO_s106
                                 + 1.1923 w_k1_MAT_s106
                                 + 1.33411 w_k1_MSFT_s106
                                 + 1.11439 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.0814 w_k1_ADM_s106
                                 + 1.26101 w_k1_AXP_s106
                                 + 1.16871 w_k1_DIS_s106
                                 + 1.25313 w_k1_FDX_s106
                                 + 0.759458 w_k1_IBM_s106
                                 + 1.05072 w_k1_INTC_s106
                                 + 1.02702 w_k1_KO_s106
                                 + 0.851506 w_k1_MAT_s106
                                 + 1.22102 w_k1_MSFT_s106
                                 + 0.952415 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.980806 w_k1_ADM_s106
                                 + 0.966048 w_k1_AXP_s106
                                 + 0.875989 w_k1_DIS_s106
                                 + 1.00035 w_k1_FDX_s106
                                 + 0.769077 w_k1_IBM_s106
                                 + 0.922021 w_k1_INTC_s106
                                 + 0.868748 w_k1_KO_s106
                                 + 0.872877 w_k1_MAT_s106
                                 + 0.946255 w_k1_MSFT_s106
                                 + 0.881513 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.855666 w_k1_ADM_s106
                                 + 0.719198 w_k1_AXP_s106
                                 + 0.765152 w_k1_DIS_s106
                                 + 0.827346 w_k1_FDX_s106
                                 + 0.73985 w_k1_IBM_s106
                                 + 0.846309 w_k1_INTC_s106
                                 + 0.866491 w_k1_KO_s106
                                 + 0.513954 w_k1_MAT_s106
                                 + 0.949188 w_k1_MSFT_s106
                                 + 0.844682 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.821961 w_k1_ADM_s106
                                 + 0.586347 w_k1_AXP_s106
                                 + 0.731001 w_k1_DIS_s106
                                 + 0.712052 w_k1_FDX_s106
                                 + 0.786057 w_k1_IBM_s106
                                 + 0.66066 w_k1_INTC_s106
                                 + 0.880122 w_k1_KO_s106
                                 + 0.729276 w_k1_MAT_s106
                                 + 0.751263 w_k1_MSFT_s106
                                 + 0.771251 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.02505 w_k1_ADM_s106
                                 + 0.773422 w_k1_AXP_s106
                                 + 1.21217 w_k1_DIS_s106
                                 + 0.964187 w_k1_FDX_s106
                                 + 1.12283 w_k1_IBM_s106
                                 + 0.876821 w_k1_INTC_s106
                                 + 1.21006 w_k1_KO_s106
                                 + 0.898201 w_k1_MAT_s106
                                 + 0.986525 w_k1_MSFT_s106
                                 + 0.996504 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.19737 w_k1_ADM_s106
                                 + 1.35632 w_k1_AXP_s106
                                 + 1.64386 w_k1_DIS_s106
                                 + 1.15224 w_k1_FDX_s106
                                 + 1.57828 w_k1_IBM_s106
                                 + 0.938896 w_k1_INTC_s106
                                 + 1.54496 w_k1_KO_s106 + 1.41558 w_k1_MAT_s106
                                 + 1.19712 w_k1_MSFT_s106
                                 + 1.15407 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.804641 w_k1_ADM_s106
                                 + 1.15251 w_k1_AXP_s106
                                 + 1.07601 w_k1_DIS_s106
                                 + 0.987877 w_k1_FDX_s106
                                 + 1.18739 w_k1_IBM_s106
                                 + 1.11276 w_k1_INTC_s106
                                 + 1.09426 w_k1_KO_s106
                                 + 0.985927 w_k1_MAT_s106
                                 + 0.927773 w_k1_MSFT_s106
                                 + 0.983738 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.43652 w_k1_ADM_s107
                                 + 1.13187 w_k1_AXP_s107
                                 + 1.11486 w_k1_DIS_s107
                                 + 1.34273 w_k1_FDX_s107
                                 + 0.719065 w_k1_IBM_s107
                                 + 1.09136 w_k1_INTC_s107
                                 + 0.831302 w_k1_KO_s107
                                 + 0.636979 w_k1_MAT_s107
                                 + 1.223 w_k1_MSFT_s107 + 1.19436 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.08518 w_k1_ADM_s107
                                 + 1.14223 w_k1_AXP_s107
                                 + 1.25078 w_k1_DIS_s107
                                 + 1.07059 w_k1_FDX_s107
                                 + 1.34071 w_k1_IBM_s107
                                 + 0.91382 w_k1_INTC_s107
                                 + 1.23625 w_k1_KO_s107 + 1.13877 w_k1_MAT_s107
                                 + 1.02535 w_k1_MSFT_s107
                                 + 1.17105 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.14092 w_k1_ADM_s107
                                 + 1.0318 w_k1_AXP_s107 + 1.06987 w_k1_DIS_s107
                                 + 1.07942 w_k1_FDX_s107
                                 + 0.972884 w_k1_IBM_s107
                                 + 0.84874 w_k1_INTC_s107
                                 + 0.818968 w_k1_KO_s107
                                 + 0.576186 w_k1_MAT_s107
                                 + 0.990122 w_k1_MSFT_s107
                                 + 1.0133 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.08322 w_k1_ADM_s107
                                 + 1.47041 w_k1_AXP_s107
                                 + 1.26476 w_k1_DIS_s107
                                 + 1.19977 w_k1_FDX_s107 + 1.1908 w_k1_IBM_s107
                                 + 1.0208 w_k1_INTC_s107
                                 + 0.936571 w_k1_KO_s107
                                 + 1.29816 w_k1_MAT_s107
                                 + 1.07709 w_k1_MSFT_s107
                                 + 1.13439 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.56679 w_k1_ADM_s107
                                 + 1.4835 w_k1_AXP_s107 + 1.54261 w_k1_DIS_s107
                                 + 1.46778 w_k1_FDX_s107
                                 + 1.29661 w_k1_IBM_s107
                                 + 1.22224 w_k1_INTC_s107
                                 + 1.30895 w_k1_KO_s107 + 1.40935 w_k1_MAT_s107
                                 + 1.37227 w_k1_MSFT_s107
                                 + 1.32668 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.04627 w_k1_ADM_s107
                                 + 0.826436 w_k1_AXP_s107
                                 + 1.35533 w_k1_DIS_s107 + 1.1942 w_k1_FDX_s107
                                 + 1.0884 w_k1_IBM_s107
                                 + 1.09495 w_k1_INTC_s107
                                 + 1.08129 w_k1_KO_s107
                                 + 0.587795 w_k1_MAT_s107
                                 + 1.30422 w_k1_MSFT_s107
                                 + 1.11905 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.780767 w_k1_ADM_s107
                                 + 0.75963 w_k1_AXP_s107
                                 + 0.765309 w_k1_DIS_s107
                                 + 0.799339 w_k1_FDX_s107
                                 + 0.994071 w_k1_IBM_s107
                                 + 0.812467 w_k1_INTC_s107
                                 + 0.729128 w_k1_KO_s107
                                 + 0.678162 w_k1_MAT_s107
                                 + 0.831867 w_k1_MSFT_s107
                                 + 0.896533 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.07154 w_k1_ADM_s107
                                 + 0.89751 w_k1_AXP_s107
                                 + 0.753906 w_k1_DIS_s107
                                 + 0.889971 w_k1_FDX_s107
                                 + 0.844889 w_k1_IBM_s107
                                 + 0.837262 w_k1_INTC_s107
                                 + 0.710759 w_k1_KO_s107
                                 + 0.645202 w_k1_MAT_s107
                                 + 0.929146 w_k1_MSFT_s107
                                 + 0.969204 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.779473 w_k1_ADM_s110
                                 + 1.10335 w_k1_AXP_s110
                                 + 0.822125 w_k1_DIS_s110
                                 + 0.938295 w_k1_FDX_s110
                                 + 0.945017 w_k1_IBM_s110
                                 + 1.04782 w_k1_INTC_s110
                                 + 0.948465 w_k1_KO_s110
                                 + 0.839923 w_k1_MAT_s110
                                 + 0.900011 w_k1_MSFT_s110
                                 + 0.915015 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.35201 w_k1_ADM_s110
                                 + 1.24677 w_k1_AXP_s110
                                 + 1.45683 w_k1_DIS_s110
                                 + 1.45134 w_k1_FDX_s110
                                 + 0.866232 w_k1_IBM_s110
                                 + 1.32993 w_k1_INTC_s110
                                 + 1.29179 w_k1_KO_s110
                                 + 0.795863 w_k1_MAT_s110
                                 + 1.35822 w_k1_MSFT_s110
                                 + 1.22863 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.826975 w_k1_ADM_s110
                                 + 1.19954 w_k1_AXP_s110
                                 + 0.95225 w_k1_DIS_s110
                                 + 1.00498 w_k1_FDX_s110
                                 + 1.53905 w_k1_IBM_s110
                                 + 0.96852 w_k1_INTC_s110
                                 + 1.16983 w_k1_KO_s110 + 1.38011 w_k1_MAT_s110
                                 + 0.813545 w_k1_MSFT_s110
                                 + 1.03021 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.12668 w_k1_ADM_s110
                                 + 0.88946 w_k1_AXP_s110
                                 + 1.22039 w_k1_DIS_s110
                                 + 1.14986 w_k1_FDX_s110
                                 + 1.23878 w_k1_IBM_s110
                                 + 1.26945 w_k1_INTC_s110
                                 + 1.30673 w_k1_KO_s110 + 1.10949 w_k1_MAT_s110
                                 + 1.31291 w_k1_MSFT_s110
                                 + 1.14113 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.953755 w_k1_ADM_s110
                                 + 1.13392 w_k1_AXP_s110
                                 + 0.951536 w_k1_DIS_s110
                                 + 1.11598 w_k1_FDX_s110
                                 + 0.723253 w_k1_IBM_s110
                                 + 0.937968 w_k1_INTC_s110
                                 + 0.942357 w_k1_KO_s110
                                 + 0.677195 w_k1_MAT_s110
                                 + 0.990661 w_k1_MSFT_s110
                                 + 1.03995 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18172 w_k1_ADM_s110
                                 + 1.3722 w_k1_AXP_s110 + 1.18704 w_k1_DIS_s110
                                 + 1.21772 w_k1_FDX_s110
                                 + 1.16068 w_k1_IBM_s110
                                 + 1.0069 w_k1_INTC_s110 + 1.12177 w_k1_KO_s110
                                 + 1.22377 w_k1_MAT_s110
                                 + 1.06094 w_k1_MSFT_s110
                                 + 1.05898 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.615199 w_k1_ADM_s110
                                 + 0.693759 w_k1_AXP_s110
                                 + 0.578281 w_k1_DIS_s110
                                 + 0.799354 w_k1_FDX_s110
                                 + 0.678862 w_k1_IBM_s110
                                 + 0.822771 w_k1_INTC_s110
                                 + 0.722135 w_k1_KO_s110
                                 + 0.783547 w_k1_MAT_s110
                                 + 0.707872 w_k1_MSFT_s110
                                 + 0.863638 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.1206 w_k1_ADM_s110
                                 + 1.49354 w_k1_AXP_s110
                                 + 1.80422 w_k1_DIS_s110
                                 + 1.50324 w_k1_FDX_s110
                                 + 1.40158 w_k1_IBM_s110
                                 + 1.2121 w_k1_INTC_s110 + 1.47862 w_k1_KO_s110
                                 + 1.71394 w_k1_MAT_s110
                                 + 1.47144 w_k1_MSFT_s110
                                 + 1.27127 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.511554 w_k1_ADM_s111
                                 + 0.832926 w_k1_AXP_s111
                                 + 1.26859 w_k1_DIS_s111
                                 + 0.97955 w_k1_FDX_s111
                                 + 0.934336 w_k1_IBM_s111
                                 + 1.31619 w_k1_INTC_s111
                                 + 1.04995 w_k1_KO_s111
                                 + 0.730018 w_k1_MAT_s111
                                 + 1.34673 w_k1_MSFT_s111
                                 + 0.928448 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.04035 w_k1_ADM_s111
                                 + 0.959157 w_k1_AXP_s111
                                 + 0.910363 w_k1_DIS_s111
                                 + 0.916728 w_k1_FDX_s111
                                 + 1.05173 w_k1_IBM_s111
                                 + 0.899901 w_k1_INTC_s111
                                 + 1.00716 w_k1_KO_s111
                                 + 0.838186 w_k1_MAT_s111
                                 + 1.08026 w_k1_MSFT_s111
                                 + 1.03135 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.49336 w_k1_ADM_s111
                                 + 1.34737 w_k1_AXP_s111
                                 + 1.38613 w_k1_DIS_s111
                                 + 1.27434 w_k1_FDX_s111
                                 + 1.14844 w_k1_IBM_s111
                                 + 1.26576 w_k1_INTC_s111
                                 + 1.30928 w_k1_KO_s111
                                 + 0.990259 w_k1_MAT_s111
                                 + 1.32051 w_k1_MSFT_s111
                                 + 1.19735 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.716135 w_k1_ADM_s111
                                 + 0.606301 w_k1_AXP_s111
                                 + 0.548913 w_k1_DIS_s111
                                 + 0.745916 w_k1_FDX_s111
                                 + 0.556895 w_k1_IBM_s111
                                 + 0.697597 w_k1_INTC_s111
                                 + 0.617802 w_k1_KO_s111
                                 + 0.552091 w_k1_MAT_s111
                                 + 0.862989 w_k1_MSFT_s111
                                 + 0.850034 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.696556 w_k1_ADM_s111
                                 + 0.718539 w_k1_AXP_s111
                                 + 0.669567 w_k1_DIS_s111
                                 + 0.766017 w_k1_FDX_s111
                                 + 0.925115 w_k1_IBM_s111
                                 + 0.891031 w_k1_INTC_s111
                                 + 0.798591 w_k1_KO_s111
                                 + 0.759986 w_k1_MAT_s111
                                 + 0.818717 w_k1_MSFT_s111
                                 + 0.880339 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.07023 w_k1_ADM_s111
                                 + 0.886178 w_k1_AXP_s111
                                 + 0.825884 w_k1_DIS_s111
                                 + 0.953094 w_k1_FDX_s111
                                 + 0.694044 w_k1_IBM_s111
                                 + 1.12448 w_k1_INTC_s111
                                 + 0.844452 w_k1_KO_s111
                                 + 0.659217 w_k1_MAT_s111
                                 + 1.01556 w_k1_MSFT_s111
                                 + 0.896721 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.891149 w_k1_ADM_s111
                                 + 0.800324 w_k1_AXP_s111
                                 + 0.954629 w_k1_DIS_s111
                                 + 0.888726 w_k1_FDX_s111
                                 + 1.24815 w_k1_IBM_s111
                                 + 0.827345 w_k1_INTC_s111
                                 + 1.16694 w_k1_KO_s111 + 1.05412 w_k1_MAT_s111
                                 + 1.05549 w_k1_MSFT_s111
                                 + 1.13126 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.12661 w_k1_ADM_s111
                                 + 1.26368 w_k1_AXP_s111
                                 + 1.20409 w_k1_DIS_s111
                                 + 1.10229 w_k1_FDX_s111
                                 + 1.54801 w_k1_IBM_s111
                                 + 1.21288 w_k1_INTC_s111
                                 + 1.48798 w_k1_KO_s111 + 1.29565 w_k1_MAT_s111
                                 + 1.16838 w_k1_MSFT_s111
                                 + 1.16749 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.14157 w_k1_ADM_s112
                                 + 1.32552 w_k1_AXP_s112
                                 + 1.04477 w_k1_DIS_s112 + 1.2021 w_k1_FDX_s112
                                 + 0.963367 w_k1_IBM_s112
                                 + 1.11106 w_k1_INTC_s112
                                 + 0.94082 w_k1_KO_s112 + 1.1404 w_k1_MAT_s112
                                 + 1.08935 w_k1_MSFT_s112
                                 + 0.977973 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.745865 w_k1_ADM_s112
                                 + 0.847009 w_k1_AXP_s112
                                 + 0.59095 w_k1_DIS_s112
                                 + 0.779883 w_k1_FDX_s112
                                 + 0.749119 w_k1_IBM_s112
                                 + 0.994429 w_k1_INTC_s112
                                 + 0.711208 w_k1_KO_s112
                                 + 0.803323 w_k1_MAT_s112
                                 + 0.785025 w_k1_MSFT_s112
                                 + 0.859758 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.986504 w_k1_ADM_s112
                                 + 1.14914 w_k1_AXP_s112
                                 + 0.893979 w_k1_DIS_s112
                                 + 0.921221 w_k1_FDX_s112
                                 + 1.13672 w_k1_IBM_s112
                                 + 0.911244 w_k1_INTC_s112
                                 + 0.870548 w_k1_KO_s112
                                 + 1.33429 w_k1_MAT_s112
                                 + 0.91256 w_k1_MSFT_s112
                                 + 0.917941 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.03716 w_k1_ADM_s112
                                 + 0.997 w_k1_AXP_s112 + 0.808942 w_k1_DIS_s112
                                 + 1.07286 w_k1_FDX_s112
                                 + 0.704316 w_k1_IBM_s112
                                 + 1.22547 w_k1_INTC_s112
                                 + 0.882825 w_k1_KO_s112
                                 + 0.923452 w_k1_MAT_s112
                                 + 1.06953 w_k1_MSFT_s112
                                 + 1.02494 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.1923 w_k1_ADM_s112
                                 + 1.11448 w_k1_AXP_s112
                                 + 1.43438 w_k1_DIS_s112
                                 + 1.17918 w_k1_FDX_s112
                                 + 1.11951 w_k1_IBM_s112
                                 + 1.27965 w_k1_INTC_s112
                                 + 1.32483 w_k1_KO_s112 + 1.07331 w_k1_MAT_s112
                                 + 1.35024 w_k1_MSFT_s112
                                 + 1.06975 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.0624 w_k1_ADM_s112
                                 + 1.01031 w_k1_AXP_s112
                                 + 0.914936 w_k1_DIS_s112
                                 + 1.02976 w_k1_FDX_s112
                                 + 0.954744 w_k1_IBM_s112
                                 + 1.04528 w_k1_INTC_s112
                                 + 1.02149 w_k1_KO_s112
                                 + 0.932028 w_k1_MAT_s112
                                 + 1.09051 w_k1_MSFT_s112
                                 + 0.98793 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.29623 w_k1_ADM_s112
                                 + 1.5808 w_k1_AXP_s112 + 1.45077 w_k1_DIS_s112
                                 + 1.24911 w_k1_FDX_s112 + 1.8049 w_k1_IBM_s112
                                 + 1.14869 w_k1_INTC_s112
                                 + 1.20747 w_k1_KO_s112 + 1.38304 w_k1_MAT_s112
                                 + 1.13311 w_k1_MSFT_s112
                                 + 1.20904 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12587 w_k1_ADM_s112
                                 + 1.7328 w_k1_AXP_s112 + 1.43395 w_k1_DIS_s112
                                 + 1.49343 w_k1_FDX_s112
                                 + 1.03713 w_k1_IBM_s112
                                 + 1.66055 w_k1_INTC_s112
                                 + 1.00376 w_k1_KO_s112 + 1.13928 w_k1_MAT_s112
                                 + 1.31363 w_k1_MSFT_s112 + 1.0167 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.19527 w_k1_ADM_s113
                                 + 0.975769 w_k1_AXP_s113
                                 + 1.20073 w_k1_DIS_s113 + 1.0821 w_k1_FDX_s113
                                 + 1.04226 w_k1_IBM_s113
                                 + 1.10425 w_k1_INTC_s113
                                 + 1.14157 w_k1_KO_s113
                                 + 0.920903 w_k1_MAT_s113
                                 + 1.18233 w_k1_MSFT_s113
                                 + 1.12675 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.00504 w_k1_ADM_s113
                                 + 0.83755 w_k1_AXP_s113
                                 + 0.990077 w_k1_DIS_s113
                                 + 1.08444 w_k1_FDX_s113
                                 + 0.804737 w_k1_IBM_s113
                                 + 1.0862 w_k1_INTC_s113
                                 + 0.954925 w_k1_KO_s113
                                 + 0.746902 w_k1_MAT_s113
                                 + 0.999792 w_k1_MSFT_s113
                                 + 1.15495 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.34837 w_k1_ADM_s113
                                 + 1.34909 w_k1_AXP_s113
                                 + 1.42392 w_k1_DIS_s113
                                 + 1.30492 w_k1_FDX_s113
                                 + 1.18352 w_k1_IBM_s113
                                 + 1.12124 w_k1_INTC_s113
                                 + 1.37107 w_k1_KO_s113 + 1.10456 w_k1_MAT_s113
                                 + 1.29469 w_k1_MSFT_s113
                                 + 1.41077 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.32754 w_k1_ADM_s113
                                 + 1.19325 w_k1_AXP_s113
                                 + 1.63618 w_k1_DIS_s113
                                 + 1.19354 w_k1_FDX_s113
                                 + 1.53313 w_k1_IBM_s113
                                 + 1.01353 w_k1_INTC_s113
                                 + 1.38705 w_k1_KO_s113 + 1.51755 w_k1_MAT_s113
                                 + 1.27398 w_k1_MSFT_s113
                                 + 1.19957 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.810366 w_k1_ADM_s113
                                 + 0.684796 w_k1_AXP_s113
                                 + 0.640831 w_k1_DIS_s113
                                 + 0.738454 w_k1_FDX_s113
                                 + 0.895647 w_k1_IBM_s113
                                 + 0.832723 w_k1_INTC_s113
                                 + 0.804043 w_k1_KO_s113
                                 + 0.978926 w_k1_MAT_s113
                                 + 0.769697 w_k1_MSFT_s113
                                 + 0.953389 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.05786 w_k1_ADM_s113
                                 + 0.892022 w_k1_AXP_s113
                                 + 0.826184 w_k1_DIS_s113
                                 + 0.872273 w_k1_FDX_s113
                                 + 0.862861 w_k1_IBM_s113
                                 + 0.942766 w_k1_INTC_s113
                                 + 0.93747 w_k1_KO_s113
                                 + 0.793592 w_k1_MAT_s113
                                 + 0.891889 w_k1_MSFT_s113
                                 + 1.03343 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.978578 w_k1_ADM_s113
                                 + 0.967357 w_k1_AXP_s113
                                 + 0.972851 w_k1_DIS_s113
                                 + 0.924247 w_k1_FDX_s113
                                 + 1.33357 w_k1_IBM_s113
                                 + 0.921116 w_k1_INTC_s113
                                 + 1.13354 w_k1_KO_s113 + 1.38451 w_k1_MAT_s113
                                 + 0.934885 w_k1_MSFT_s113
                                 + 1.08065 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.832463 w_k1_ADM_s113
                                 + 0.563875 w_k1_AXP_s113
                                 + 0.672374 w_k1_DIS_s113
                                 + 0.765275 w_k1_FDX_s113
                                 + 0.656324 w_k1_IBM_s113
                                 + 0.831484 w_k1_INTC_s113
                                 + 0.705772 w_k1_KO_s113
                                 + 0.565193 w_k1_MAT_s113
                                 + 0.861131 w_k1_MSFT_s113
                                 + 0.916846 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.18018 w_k1_ADM_s114
                                 + 1.24501 w_k1_AXP_s114
                                 + 0.943862 w_k1_DIS_s114
                                 + 1.14661 w_k1_FDX_s114
                                 + 0.886622 w_k1_IBM_s114
                                 + 0.907122 w_k1_INTC_s114
                                 + 0.802364 w_k1_KO_s114 + 0.9713 w_k1_MAT_s114
                                 + 1.07027 w_k1_MSFT_s114
                                 + 0.844217 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.4815 w_k1_ADM_s114
                                 + 1.41187 w_k1_AXP_s114
                                 + 0.87388 w_k1_DIS_s114
                                 + 1.05689 w_k1_FDX_s114
                                 + 1.15414 w_k1_IBM_s114
                                 + 1.04352 w_k1_INTC_s114
                                 + 0.951643 w_k1_KO_s114
                                 + 1.20252 w_k1_MAT_s114
                                 + 1.02359 w_k1_MSFT_s114
                                 + 0.939206 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.914708 w_k1_ADM_s114
                                 + 0.688284 w_k1_AXP_s114
                                 + 0.820605 w_k1_DIS_s114
                                 + 0.875327 w_k1_FDX_s114
                                 + 0.949627 w_k1_IBM_s114
                                 + 0.879169 w_k1_INTC_s114
                                 + 0.823803 w_k1_KO_s114
                                 + 0.609041 w_k1_MAT_s114
                                 + 0.935068 w_k1_MSFT_s114
                                 + 0.859878 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.42173 w_k1_ADM_s114
                                 + 1.43247 w_k1_AXP_s114
                                 + 1.81404 w_k1_DIS_s114
                                 + 1.52029 w_k1_FDX_s114
                                 + 1.51189 w_k1_IBM_s114
                                 + 1.40071 w_k1_INTC_s114 + 1.4784 w_k1_KO_s114
                                 + 1.37045 w_k1_MAT_s114
                                 + 1.47654 w_k1_MSFT_s114
                                 + 1.09373 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.927435 w_k1_ADM_s114
                                 + 0.745261 w_k1_AXP_s114
                                 + 0.754466 w_k1_DIS_s114
                                 + 0.760968 w_k1_FDX_s114
                                 + 1.0585 w_k1_IBM_s114
                                 + 0.890575 w_k1_INTC_s114
                                 + 0.943907 w_k1_KO_s114
                                 + 0.859043 w_k1_MAT_s114
                                 + 0.977352 w_k1_MSFT_s114
                                 + 0.93731 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.73098 w_k1_ADM_s114
                                 + 0.711354 w_k1_AXP_s114
                                 + 0.562721 w_k1_DIS_s114
                                 + 0.720954 w_k1_FDX_s114
                                 + 0.646972 w_k1_IBM_s114
                                 + 0.662562 w_k1_INTC_s114
                                 + 0.57661 w_k1_KO_s114
                                 + 0.560033 w_k1_MAT_s114
                                 + 0.717129 w_k1_MSFT_s114
                                 + 0.692487 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.15457 w_k1_ADM_s114
                                 + 0.987867 w_k1_AXP_s114
                                 + 1.15114 w_k1_DIS_s114
                                 + 1.05324 w_k1_FDX_s114
                                 + 1.28415 w_k1_IBM_s114
                                 + 1.11427 w_k1_INTC_s114
                                 + 1.22826 w_k1_KO_s114
                                 + 0.930074 w_k1_MAT_s114
                                 + 1.19495 w_k1_MSFT_s114
                                 + 1.00767 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.10156 w_k1_ADM_s114
                                 + 0.855748 w_k1_AXP_s114
                                 + 0.903737 w_k1_DIS_s114
                                 + 1.04573 w_k1_FDX_s114
                                 + 0.686679 w_k1_IBM_s114
                                 + 0.984645 w_k1_INTC_s114
                                 + 0.808299 w_k1_KO_s114
                                 + 0.637756 w_k1_MAT_s114
                                 + 0.906182 w_k1_MSFT_s114
                                 + 0.84606 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.828858 w_k1_ADM_s115
                                 + 0.695081 w_k1_AXP_s115
                                 + 0.719415 w_k1_DIS_s115
                                 + 0.829222 w_k1_FDX_s115
                                 + 0.837551 w_k1_IBM_s115
                                 + 0.81727 w_k1_INTC_s115
                                 + 0.801019 w_k1_KO_s115
                                 + 0.866123 w_k1_MAT_s115
                                 + 0.793532 w_k1_MSFT_s115
                                 + 0.855563 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.883823 w_k1_ADM_s115
                                 + 0.849152 w_k1_AXP_s115
                                 + 0.745228 w_k1_DIS_s115
                                 + 0.789501 w_k1_FDX_s115
                                 + 1.15021 w_k1_IBM_s115
                                 + 0.972802 w_k1_INTC_s115
                                 + 0.952646 w_k1_KO_s115
                                 + 1.21715 w_k1_MAT_s115
                                 + 0.872896 w_k1_MSFT_s115
                                 + 0.877128 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.3606 w_k1_ADM_s115
                                 + 1.38214 w_k1_AXP_s115
                                 + 1.30713 w_k1_DIS_s115
                                 + 1.40616 w_k1_FDX_s115
                                 + 1.32758 w_k1_IBM_s115
                                 + 1.4838 w_k1_INTC_s115 + 1.45748 w_k1_KO_s115
                                 + 1.49859 w_k1_MAT_s115
                                 + 1.31848 w_k1_MSFT_s115
                                 + 1.17734 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.46245 w_k1_ADM_s115
                                 + 1.32782 w_k1_AXP_s115
                                 + 1.26245 w_k1_DIS_s115
                                 + 1.20307 w_k1_FDX_s115
                                 + 0.954808 w_k1_IBM_s115
                                 + 1.14483 w_k1_INTC_s115
                                 + 1.10898 w_k1_KO_s115 + 1.3174 w_k1_MAT_s115
                                 + 1.22183 w_k1_MSFT_s115
                                 + 1.05184 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09512 w_k1_ADM_s115
                                 + 1.00855 w_k1_AXP_s115
                                 + 1.09561 w_k1_DIS_s115
                                 + 0.978026 w_k1_FDX_s115
                                 + 1.43452 w_k1_IBM_s115
                                 + 1.16794 w_k1_INTC_s115
                                 + 1.31659 w_k1_KO_s115 + 1.3785 w_k1_MAT_s115
                                 + 1.15378 w_k1_MSFT_s115
                                 + 1.05975 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.945885 w_k1_ADM_s115
                                 + 1.03235 w_k1_AXP_s115
                                 + 1.39597 w_k1_DIS_s115
                                 + 1.23217 w_k1_FDX_s115
                                 + 1.02733 w_k1_IBM_s115
                                 + 1.3527 w_k1_INTC_s115 + 1.24941 w_k1_KO_s115
                                 + 0.934931 w_k1_MAT_s115
                                 + 1.38496 w_k1_MSFT_s115
                                 + 1.15595 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.01803 w_k1_ADM_s115
                                 + 0.91863 w_k1_AXP_s115
                                 + 0.738233 w_k1_DIS_s115
                                 + 0.950005 w_k1_FDX_s115
                                 + 0.672916 w_k1_IBM_s115
                                 + 0.929329 w_k1_INTC_s115
                                 + 0.789929 w_k1_KO_s115
                                 + 0.933287 w_k1_MAT_s115
                                 + 0.933067 w_k1_MSFT_s115
                                 + 0.761672 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.13232 w_k1_ADM_s115
                                 + 0.918864 w_k1_AXP_s115
                                 + 0.941056 w_k1_DIS_s115
                                 + 0.968269 w_k1_FDX_s115
                                 + 1.04763 w_k1_IBM_s115
                                 + 1.00358 w_k1_INTC_s115
                                 + 1.06854 w_k1_KO_s115 + 1.10688 w_k1_MAT_s115
                                 + 0.934438 w_k1_MSFT_s115
                                 + 0.991812 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.28998 w_k1_ADM_s116
                                 + 1.14314 w_k1_AXP_s116
                                 + 1.54664 w_k1_DIS_s116
                                 + 1.35751 w_k1_FDX_s116
                                 + 0.796216 w_k1_IBM_s116
                                 + 1.14436 w_k1_INTC_s116
                                 + 0.998846 w_k1_KO_s116
                                 + 0.918382 w_k1_MAT_s116
                                 + 1.30364 w_k1_MSFT_s116
                                 + 1.20296 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.10995 w_k1_ADM_s116
                                 + 1.00788 w_k1_AXP_s116
                                 + 0.872886 w_k1_DIS_s116
                                 + 0.953247 w_k1_FDX_s116
                                 + 1.11711 w_k1_IBM_s116
                                 + 0.7472 w_k1_INTC_s116
                                 + 0.968044 w_k1_KO_s116
                                 + 1.24972 w_k1_MAT_s116
                                 + 0.911851 w_k1_MSFT_s116
                                 + 1.09549 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.6649 w_k1_ADM_s116
                                 + 1.52331 w_k1_AXP_s116
                                 + 1.45579 w_k1_DIS_s116
                                 + 1.38181 w_k1_FDX_s116
                                 + 1.18313 w_k1_IBM_s116
                                 + 1.03627 w_k1_INTC_s116
                                 + 1.21027 w_k1_KO_s116 + 1.41049 w_k1_MAT_s116
                                 + 1.20343 w_k1_MSFT_s116
                                 + 1.41684 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.06753 w_k1_ADM_s116
                                 + 1.01285 w_k1_AXP_s116
                                 + 0.877854 w_k1_DIS_s116
                                 + 1.10163 w_k1_FDX_s116
                                 + 0.701741 w_k1_IBM_s116
                                 + 0.970856 w_k1_INTC_s116
                                 + 0.868816 w_k1_KO_s116
                                 + 0.674713 w_k1_MAT_s116
                                 + 0.862723 w_k1_MSFT_s116
                                 + 1.03283 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04662 w_k1_ADM_s116
                                 + 1.06947 w_k1_AXP_s116
                                 + 1.09249 w_k1_DIS_s116
                                 + 1.17108 w_k1_FDX_s116
                                 + 1.02874 w_k1_IBM_s116
                                 + 0.956581 w_k1_INTC_s116
                                 + 1.01946 w_k1_KO_s116 + 1.47857 w_k1_MAT_s116
                                 + 1.02409 w_k1_MSFT_s116
                                 + 1.01479 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.14947 w_k1_ADM_s116
                                 + 1.24421 w_k1_AXP_s116
                                 + 1.20307 w_k1_DIS_s116
                                 + 1.08743 w_k1_FDX_s116
                                 + 1.58865 w_k1_IBM_s116
                                 + 0.791849 w_k1_INTC_s116
                                 + 1.25966 w_k1_KO_s116 + 2.14524 w_k1_MAT_s116
                                 + 0.883353 w_k1_MSFT_s116
                                 + 1.29394 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.736686 w_k1_ADM_s116
                                 + 0.703739 w_k1_AXP_s116
                                 + 0.754161 w_k1_DIS_s116
                                 + 0.711367 w_k1_FDX_s116
                                 + 1.30732 w_k1_IBM_s116
                                 + 0.694828 w_k1_INTC_s116
                                 + 0.901138 w_k1_KO_s116
                                 + 1.50281 w_k1_MAT_s116
                                 + 0.775527 w_k1_MSFT_s116
                                 + 0.97103 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.891039 w_k1_ADM_s116
                                 + 0.633001 w_k1_AXP_s116
                                 + 0.816117 w_k1_DIS_s116
                                 + 0.772595 w_k1_FDX_s116
                                 + 0.821339 w_k1_IBM_s116
                                 + 0.726153 w_k1_INTC_s116
                                 + 0.777375 w_k1_KO_s116
                                 + 0.536932 w_k1_MAT_s116
                                 + 0.86598 w_k1_MSFT_s116
                                 + 0.867436 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.84973 w_k1_ADM_s117
                                 + 0.958491 w_k1_AXP_s117
                                 + 1.01228 w_k1_DIS_s117
                                 + 0.911919 w_k1_FDX_s117
                                 + 1.24074 w_k1_IBM_s117
                                 + 0.96121 w_k1_INTC_s117
                                 + 1.35641 w_k1_KO_s117
                                 + 0.998271 w_k1_MAT_s117
                                 + 1.07709 w_k1_MSFT_s117
                                 + 1.03032 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.727882 w_k1_ADM_s117
                                 + 0.750511 w_k1_AXP_s117
                                 + 0.77577 w_k1_DIS_s117
                                 + 0.790387 w_k1_FDX_s117
                                 + 0.82409 w_k1_IBM_s117
                                 + 0.863462 w_k1_INTC_s117
                                 + 0.883673 w_k1_KO_s117
                                 + 0.573646 w_k1_MAT_s117
                                 + 0.866725 w_k1_MSFT_s117
                                 + 0.869498 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.958712 w_k1_ADM_s117
                                 + 0.95431 w_k1_AXP_s117
                                 + 1.06346 w_k1_DIS_s117
                                 + 0.986919 w_k1_FDX_s117
                                 + 0.886411 w_k1_IBM_s117
                                 + 1.03031 w_k1_INTC_s117
                                 + 1.05648 w_k1_KO_s117
                                 + 0.849964 w_k1_MAT_s117
                                 + 0.99753 w_k1_MSFT_s117
                                 + 1.08163 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.807205 w_k1_ADM_s117
                                 + 1.12169 w_k1_AXP_s117
                                 + 1.20132 w_k1_DIS_s117
                                 + 1.17065 w_k1_FDX_s117
                                 + 0.95719 w_k1_IBM_s117
                                 + 1.10678 w_k1_INTC_s117
                                 + 1.15233 w_k1_KO_s117
                                 + 0.935129 w_k1_MAT_s117
                                 + 1.0827 w_k1_MSFT_s117 + 1.12752 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.7441 w_k1_ADM_s117
                                 + 0.609384 w_k1_AXP_s117
                                 + 0.609993 w_k1_DIS_s117
                                 + 0.746083 w_k1_FDX_s117
                                 + 0.648994 w_k1_IBM_s117
                                 + 0.812979 w_k1_INTC_s117
                                 + 0.782504 w_k1_KO_s117
                                 + 0.442154 w_k1_MAT_s117
                                 + 0.815724 w_k1_MSFT_s117
                                 + 0.866996 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06692 w_k1_ADM_s117
                                 + 1.40768 w_k1_AXP_s117
                                 + 1.37603 w_k1_DIS_s117
                                 + 1.17353 w_k1_FDX_s117
                                 + 1.52224 w_k1_IBM_s117
                                 + 1.46771 w_k1_INTC_s117
                                 + 1.39884 w_k1_KO_s117 + 1.14843 w_k1_MAT_s117
                                 + 1.14088 w_k1_MSFT_s117
                                 + 1.13053 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.945788 w_k1_ADM_s117
                                 + 0.79991 w_k1_AXP_s117
                                 + 0.99332 w_k1_DIS_s117
                                 + 0.82234 w_k1_FDX_s117
                                 + 1.14011 w_k1_IBM_s117
                                 + 0.878781 w_k1_INTC_s117
                                 + 1.14008 w_k1_KO_s117
                                 + 0.919063 w_k1_MAT_s117
                                 + 1.03309 w_k1_MSFT_s117
                                 + 0.988782 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.821428 w_k1_ADM_s117
                                 + 0.999257 w_k1_AXP_s117
                                 + 0.698224 w_k1_DIS_s117
                                 + 0.929022 w_k1_FDX_s117
                                 + 0.890575 w_k1_IBM_s117
                                 + 0.986907 w_k1_INTC_s117
                                 + 0.851345 w_k1_KO_s117
                                 + 0.706119 w_k1_MAT_s117
                                 + 0.832348 w_k1_MSFT_s117
                                 + 0.930116 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0138772609199566 w_k1_ADM_s000
                                 + 0.0513757358506242 w_k1_ADM_s001
                                 + 0.0267902370810605 w_k1_ADM_s002
                                 + 0.0224675409436528 w_k1_ADM_s003
                                 + 0.0342015499433503 w_k1_ADM_s004
                                 + 0.0633890104535946 w_k1_ADM_s005
                                 + 0.0231609612174987 w_k1_ADM_s006
                                 + 0.0364976726316117 w_k1_ADM_s007
                                 + 0.0562070372845304 w_k1_ADM_s010
                                 + 0.043951246737653 w_k1_ADM_s011
                                 + 0.0125088478375069 w_k1_ADM_s012
                                 + 0.0207589051606816 w_k1_ADM_s013
                                 + 0.036213193739001 w_k1_ADM_s014
                                 + 0.0525043018756571 w_k1_ADM_s015
                                 + 0.0176604125247779 w_k1_ADM_s016
                                 + 0.0555431370346811 w_k1_ADM_s017
                                 + 0.0225728222687456 w_k1_ADM_s100
                                 + 0.047589108077789 w_k1_ADM_s101
                                 + 0.0691387586240659 w_k1_ADM_s102
                                 + 0.021085618944354 w_k1_ADM_s103
                                 + 0.0248924783587968 w_k1_ADM_s104
                                 + 0.0555377652183038 w_k1_ADM_s105
                                 + 0.0239379467560977 w_k1_ADM_s106
                                 + 0.0414325802982632 w_k1_ADM_s107
                                 + 0.01212902912465 w_k1_ADM_s110
                                 + 0.00999238627301926 w_k1_ADM_s111
                                 + 0.00803233583368459 w_k1_ADM_s112
                                 + 0.0231924843855644 w_k1_ADM_s113
                                 + 0.0126321294921144 w_k1_ADM_s114
                                 + 0.00877482902699164 w_k1_ADM_s115
                                 + 0.0399839949271591 w_k1_ADM_s116
                                 + 0.0127482048358918 w_k1_ADM_s117
                                 + 0.0171072443153595 w_k1_AXP_s000
                                 + 0.0458309776876549 w_k1_AXP_s001
                                 + 0.0325091532433703 w_k1_AXP_s002
                                 + 0.0241113356385986 w_k1_AXP_s003
                                 + 0.0368560297214221 w_k1_AXP_s004
                                 + 0.0695666619743372 w_k1_AXP_s005
                                 + 0.021313003451859 w_k1_AXP_s006
                                 + 0.0391482119494353 w_k1_AXP_s007
                                 + 0.0571290584853083 w_k1_AXP_s010
                                 + 0.047569795499114 w_k1_AXP_s011
                                 + 0.0124838977691854 w_k1_AXP_s012
                                 + 0.0198000270756235 w_k1_AXP_s013
                                 + 0.0392412363114314 w_k1_AXP_s014
                                 + 0.0538297403914893 w_k1_AXP_s015
                                 + 0.0188971796862764 w_k1_AXP_s016
                                 + 0.0553791452425197 w_k1_AXP_s017
                                 + 0.0234418469258256 w_k1_AXP_s100
                                 + 0.0538176648761588 w_k1_AXP_s101
                                 + 0.0627660394512502 w_k1_AXP_s102
                                 + 0.0229204334937955 w_k1_AXP_s103
                                 + 0.021796532517119 w_k1_AXP_s104
                                 + 0.0552130671206171 w_k1_AXP_s105
                                 + 0.0252006172287592 w_k1_AXP_s106
                                 + 0.0389156967121922 w_k1_AXP_s107
                                 + 0.0146099200150376 w_k1_AXP_s110
                                 + 0.00941967214069219 w_k1_AXP_s111
                                 + 0.00878274924573002 w_k1_AXP_s112
                                 + 0.0200686581759867 w_k1_AXP_s113
                                 + 0.0113930390858188 w_k1_AXP_s114
                                 + 0.00807777222924306 w_k1_AXP_s115
                                 + 0.0370552123780744 w_k1_AXP_s116
                                 + 0.014815426259953 w_k1_AXP_s117
                                 + 0.0162819145050968 w_k1_DIS_s000
                                 + 0.046667804381061 w_k1_DIS_s001
                                 + 0.0304786907875513 w_k1_DIS_s002
                                 + 0.0262053043096573 w_k1_DIS_s003
                                 + 0.0371517138123469 w_k1_DIS_s004
                                 + 0.0629325420386716 w_k1_DIS_s005
                                 + 0.0218882695668732 w_k1_DIS_s006
                                 + 0.0386307319365192 w_k1_DIS_s007
                                 + 0.0549084815766667 w_k1_DIS_s010
                                 + 0.047257319166022 w_k1_DIS_s011
                                 + 0.0130603684123392 w_k1_DIS_s012
                                 + 0.0208071786751694 w_k1_DIS_s013
                                 + 0.042328482921633 w_k1_DIS_s014
                                 + 0.063710806172229 w_k1_DIS_s015
                                 + 0.0177969022220768 w_k1_DIS_s016
                                 + 0.0555427990849547 w_k1_DIS_s017
                                 + 0.0234839373588808 w_k1_DIS_s100
                                 + 0.0531914331908977 w_k1_DIS_s101
                                 + 0.0695887209238954 w_k1_DIS_s102
                                 + 0.02222953505809 w_k1_DIS_s103
                                 + 0.0240351587409408 w_k1_DIS_s104
                                 + 0.0530642438978691 w_k1_DIS_s105
                                 + 0.0270419245422268 w_k1_DIS_s106
                                 + 0.039784757797648 w_k1_DIS_s107
                                 + 0.0133724971694331 w_k1_DIS_s110
                                 + 0.00964425345599124 w_k1_DIS_s111
                                 + 0.00756277173012493 w_k1_DIS_s112
                                 + 0.0222725534605976 w_k1_DIS_s113
                                 + 0.0113010661830857 w_k1_DIS_s114
                                 + 0.0080838515910975 w_k1_DIS_s115
                                 + 0.0370216601174872 w_k1_DIS_s116
                                 + 0.0153016579339354 w_k1_DIS_s117
                                 + 0.0162641788613407 w_k1_FDX_s000
                                 + 0.0485461689928697 w_k1_FDX_s001
                                 + 0.0304404955883937 w_k1_FDX_s002
                                 + 0.0253942385159423 w_k1_FDX_s003
                                 + 0.0352982958227213 w_k1_FDX_s004
                                 + 0.0632832134258553 w_k1_FDX_s005
                                 + 0.0209813803636404 w_k1_FDX_s006
                                 + 0.0383798455234435 w_k1_FDX_s007
                                 + 0.0563121491023075 w_k1_FDX_s010
                                 + 0.0435805614536031 w_k1_FDX_s011
                                 + 0.0131376247562824 w_k1_FDX_s012
                                 + 0.0192223661081795 w_k1_FDX_s013
                                 + 0.04140895430805 w_k1_FDX_s014
                                 + 0.0602089706860138 w_k1_FDX_s015
                                 + 0.0190895348211505 w_k1_FDX_s016
                                 + 0.055299355181751 w_k1_FDX_s017
                                 + 0.0220998502873265 w_k1_FDX_s100
                                 + 0.0506435753348927 w_k1_FDX_s101
                                 + 0.0667881768007441 w_k1_FDX_s102
                                 + 0.0216862996010549 w_k1_FDX_s103
                                 + 0.0223251769230673 w_k1_FDX_s104
                                 + 0.0526180035091197 w_k1_FDX_s105
                                 + 0.0256009178360796 w_k1_FDX_s106
                                 + 0.0397972728693751 w_k1_FDX_s107
                                 + 0.0141382281959802 w_k1_FDX_s110
                                 + 0.00950133234966812 w_k1_FDX_s111
                                 + 0.00809013053232093 w_k1_FDX_s112
                                 + 0.0213362426525523 w_k1_FDX_s113
                                 + 0.0117379224719929 w_k1_FDX_s114
                                 + 0.00836704491530328 w_k1_FDX_s115
                                 + 0.0378520528922959 w_k1_FDX_s116
                                 + 0.0145355154282163 w_k1_FDX_s117
                                 + 0.0176567824733114 w_k1_IBM_s000
                                 + 0.048440389958402 w_k1_IBM_s001
                                 + 0.0328044133853781 w_k1_IBM_s002
                                 + 0.0250988897798821 w_k1_IBM_s003
                                 + 0.0371857985647386 w_k1_IBM_s004
                                 + 0.063623355325086 w_k1_IBM_s005
                                 + 0.0215514340682862 w_k1_IBM_s006
                                 + 0.0391881496573494 w_k1_IBM_s007
                                 + 0.0608493762213083 w_k1_IBM_s010
                                 + 0.0458217244108102 w_k1_IBM_s011
                                 + 0.0137473264615196 w_k1_IBM_s012
                                 + 0.0194302235680306 w_k1_IBM_s013
                                 + 0.0409258623537116 w_k1_IBM_s014
                                 + 0.0592752126679522 w_k1_IBM_s015
                                 + 0.0189977105711702 w_k1_IBM_s016
                                 + 0.0559366932112671 w_k1_IBM_s017
                                 + 0.0232413150264963 w_k1_IBM_s100
                                 + 0.0537621132751797 w_k1_IBM_s101
                                 + 0.0694783581711888 w_k1_IBM_s102
                                 + 0.0231626363419624 w_k1_IBM_s103
                                 + 0.0233044169093175 w_k1_IBM_s104
                                 + 0.0521354237393493 w_k1_IBM_s105
                                 + 0.0243237136215127 w_k1_IBM_s106
                                 + 0.0398932136910687 w_k1_IBM_s107
                                 + 0.0134553859827537 w_k1_IBM_s110
                                 + 0.0100100155073209 w_k1_IBM_s111
                                 + 0.00794020414360472 w_k1_IBM_s112
                                 + 0.0216051551352945 w_k1_IBM_s113
                                 + 0.01186835566801 w_k1_IBM_s114
                                 + 0.00825159389253446 w_k1_IBM_s115
                                 + 0.0379624524136561 w_k1_IBM_s116
                                 + 0.015409585350053 w_k1_IBM_s117
                                 + 0.0160013868434099 w_k1_INTC_s000
                                 + 0.0442078231905605 w_k1_INTC_s001
                                 + 0.0327016607522791 w_k1_INTC_s002
                                 + 0.0281010156447891 w_k1_INTC_s003
                                 + 0.0338674141104976 w_k1_INTC_s004
                                 + 0.0661248995125481 w_k1_INTC_s005
                                 + 0.0251105924472147 w_k1_INTC_s006
                                 + 0.0371703650452855 w_k1_INTC_s007
                                 + 0.0604328176937705 w_k1_INTC_s010
                                 + 0.0567830690685371 w_k1_INTC_s011
                                 + 0.0126336211266715 w_k1_INTC_s012
                                 + 0.0179009004630906 w_k1_INTC_s013
                                 + 0.0386608714595774 w_k1_INTC_s014
                                 + 0.0616840967374149 w_k1_INTC_s015
                                 + 0.0158377978938974 w_k1_INTC_s016
                                 + 0.0560863300708363 w_k1_INTC_s017
                                 + 0.0254793924122675 w_k1_INTC_s100
                                 + 0.0546094021019639 w_k1_INTC_s101
                                 + 0.0709759090395773 w_k1_INTC_s102
                                 + 0.02060093852948 w_k1_INTC_s103
                                 + 0.0220623420139485 w_k1_INTC_s104
                                 + 0.047941082716154 w_k1_INTC_s105
                                 + 0.024075286466611 w_k1_INTC_s106
                                 + 0.035121045494868 w_k1_INTC_s107
                                 + 0.0135494294120808 w_k1_INTC_s110
                                 + 0.00998135118316659 w_k1_INTC_s111
                                 + 0.00859376083005225 w_k1_INTC_s112
                                 + 0.0213443375704654 w_k1_INTC_s113
                                 + 0.0113086630591925 w_k1_INTC_s114
                                 + 0.00872109296161309 w_k1_INTC_s115
                                 + 0.031534360978498 w_k1_INTC_s116
                                 + 0.0156288982033222 w_k1_INTC_s117
                                 + 0.0175451248354664 w_k1_KO_s000
                                 + 0.0498561560644393 w_k1_KO_s001
                                 + 0.0282194805261243 w_k1_KO_s002
                                 + 0.0256458674441279 w_k1_KO_s003
                                 + 0.0358879273837576 w_k1_KO_s004
                                 + 0.0605416333301942 w_k1_KO_s005
                                 + 0.0231174899934975 w_k1_KO_s006
                                 + 0.0348762422511288 w_k1_KO_s007
                                 + 0.0612476365725784 w_k1_KO_s010
                                 + 0.0463079801431417 w_k1_KO_s011
                                 + 0.0130760095815283 w_k1_KO_s012
                                 + 0.0183625475500864 w_k1_KO_s013
                                 + 0.0440387078192853 w_k1_KO_s014
                                 + 0.0577003523133916 w_k1_KO_s015
                                 + 0.0172677566783148 w_k1_KO_s016
                                 + 0.0534717935050673 w_k1_KO_s017
                                 + 0.0221717115063314 w_k1_KO_s100
                                 + 0.0504529250211313 w_k1_KO_s101
                                 + 0.0760909142050022 w_k1_KO_s102
                                 + 0.0220812804339114 w_k1_KO_s103
                                 + 0.0250637521009301 w_k1_KO_s104
                                 + 0.0506237355592746 w_k1_KO_s105
                                 + 0.0266173752601634 w_k1_KO_s106
                                 + 0.0346832703008259 w_k1_KO_s107
                                 + 0.0137946526487216 w_k1_KO_s110
                                 + 0.0102749016289151 w_k1_KO_s111
                                 + 0.00749163178568956 w_k1_KO_s112
                                 + 0.0225345157374888 w_k1_KO_s113
                                 + 0.011112312165863 w_k1_KO_s114
                                 + 0.00858765632073294 w_k1_KO_s115
                                 + 0.0358120206841235 w_k1_KO_s116
                                 + 0.0164274866885331 w_k1_KO_s117
                                 + 0.0215011177332869 w_k1_MAT_s000
                                 + 0.0529179512164973 w_k1_MAT_s001
                                 + 0.024086791293146 w_k1_MAT_s002
                                 + 0.0231414924592705 w_k1_MAT_s003
                                 + 0.0426915681678614 w_k1_MAT_s004
                                 + 0.0619653674106484 w_k1_MAT_s005
                                 + 0.0193973686252156 w_k1_MAT_s006
                                 + 0.0349281367834482 w_k1_MAT_s007
                                 + 0.0707068903599146 w_k1_MAT_s010
                                 + 0.0461693497996576 w_k1_MAT_s011
                                 + 0.00928979350372106 w_k1_MAT_s012
                                 + 0.0189180582015231 w_k1_MAT_s013
                                 + 0.0440734562770846 w_k1_MAT_s014
                                 + 0.0664295892304243 w_k1_MAT_s015
                                 + 0.0200637265634532 w_k1_MAT_s016
                                 + 0.0468199042451175 w_k1_MAT_s017
                                 + 0.0229091157324486 w_k1_MAT_s100
                                 + 0.0665946472238094 w_k1_MAT_s101
                                 + 0.0661988308443462 w_k1_MAT_s102
                                 + 0.0286791945683126 w_k1_MAT_s103
                                 + 0.0294067486989114 w_k1_MAT_s104
                                 + 0.0486618977044767 w_k1_MAT_s105
                                 + 0.022352649636321 w_k1_MAT_s106
                                 + 0.0310846764960684 w_k1_MAT_s107
                                 + 0.0134263584526637 w_k1_MAT_s110
                                 + 0.0085175543699464 w_k1_MAT_s111
                                 + 0.00839247725009039 w_k1_MAT_s112
                                 + 0.0204464056348281 w_k1_MAT_s113
                                 + 0.010003926229116 w_k1_MAT_s114
                                 + 0.00913563798676265 w_k1_MAT_s115
                                 + 0.042481854272006 w_k1_MAT_s116
                                 + 0.0126483360471251 w_k1_MAT_s117
                                 + 0.0159724777306352 w_k1_MSFT_s000
                                 + 0.0470266000496413 w_k1_MSFT_s001
                                 + 0.0318444557126141 w_k1_MSFT_s002
                                 + 0.0236122807300105 w_k1_MSFT_s003
                                 + 0.0347212540383489 w_k1_MSFT_s004
                                 + 0.060668321910628 w_k1_MSFT_s005
                                 + 0.0235221806334421 w_k1_MSFT_s006
                                 + 0.0393286745434161 w_k1_MSFT_s007
                                 + 0.0595707830603153 w_k1_MSFT_s010
                                 + 0.0480460168759717 w_k1_MSFT_s011
                                 + 0.0129685970831401 w_k1_MSFT_s012
                                 + 0.0192455230054031 w_k1_MSFT_s013
                                 + 0.0407960228666299 w_k1_MSFT_s014
                                 + 0.0635225119785491 w_k1_MSFT_s015
                                 + 0.0184533569411432 w_k1_MSFT_s016
                                 + 0.0532213162502462 w_k1_MSFT_s017
                                 + 0.0238512055232685 w_k1_MSFT_s100
                                 + 0.0481662931906245 w_k1_MSFT_s101
                                 + 0.0698831994566041 w_k1_MSFT_s102
                                 + 0.0222047622797843 w_k1_MSFT_s103
                                 + 0.022865161708346 w_k1_MSFT_s104
                                 + 0.0522293211870475 w_k1_MSFT_s105
                                 + 0.0254263635953453 w_k1_MSFT_s106
                                 + 0.0389433655388186 w_k1_MSFT_s107
                                 + 0.0131502784725612 w_k1_MSFT_s110
                                 + 0.0105227110341661 w_k1_MSFT_s111
                                 + 0.00811779792721433 w_k1_MSFT_s112
                                 + 0.0220410431086403 w_k1_MSFT_s113
                                 + 0.0120888031795909 w_k1_MSFT_s114
                                 + 0.00849828394347617 w_k1_MSFT_s115
                                 + 0.0341508703085171 w_k1_MSFT_s116
                                 + 0.0148300591454019 w_k1_MSFT_s117
                                 + 0.0171479716166908 w_k1_PG_s000
                                 + 0.0488798077505855 w_k1_PG_s001
                                 + 0.0306442711019104 w_k1_PG_s002
                                 + 0.023327326957445 w_k1_PG_s003
                                 + 0.0364302202247972 w_k1_PG_s004
                                 + 0.0638020516050628 w_k1_PG_s005
                                 + 0.0197201954952374 w_k1_PG_s006
                                 + 0.0374542842394625 w_k1_PG_s007
                                 + 0.0575410065180361 w_k1_PG_s010
                                 + 0.0458630619146985 w_k1_PG_s011
                                 + 0.0120309681151333 w_k1_PG_s012
                                 + 0.018704601969615 w_k1_PG_s013
                                 + 0.0398254019280246 w_k1_PG_s014
                                 + 0.0614579086534601 w_k1_PG_s015
                                 + 0.0185218737407809 w_k1_PG_s016
                                 + 0.0551418439544878 w_k1_PG_s017
                                 + 0.0215111155204589 w_k1_PG_s100
                                 + 0.0501167904603237 w_k1_PG_s101
                                 + 0.0687954623292505 w_k1_PG_s102
                                 + 0.0216375973128713 w_k1_PG_s103
                                 + 0.0231155353488806 w_k1_PG_s104
                                 + 0.0498275759042297 w_k1_PG_s105
                                 + 0.0236385016498036 w_k1_PG_s106
                                 + 0.0402378195852735 w_k1_PG_s107
                                 + 0.0133460477718307 w_k1_PG_s110
                                 + 0.0100012225069469 w_k1_PG_s111
                                 + 0.0076891452504748 w_k1_PG_s112
                                 + 0.023611185895071 w_k1_PG_s113
                                 + 0.0104065569925117 w_k1_PG_s114
                                 + 0.00788641106461603 w_k1_PG_s115
                                 + 0.0398278439561836 w_k1_PG_s116
                                 + 0.0150987013178292 w_k1_PG_s117 >= 1.12
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
