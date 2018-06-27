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
 w_k1_ADM_s0_return_constraint:  1.16141 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.91912 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.03051 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.796336 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.18463 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.890776 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.15922 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.86886 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.838124 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.21575 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.28476 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.879007 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.10718 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.776215 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.84976 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.817132 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.01698 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.924311 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.12443 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.848842 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.813262 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.03612 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.01772 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.853307 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.995808 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.17365 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.05436 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.41108 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.06783 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.719288 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05825 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.938096 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.852569 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.82417 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.37187 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.02392 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.13398 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.31212 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2013 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.894124 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.16818 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.730439 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19453 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.741758 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.33695 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.841164 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.869721 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.939207 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.30267 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.736423 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.886991 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.586107 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.729577 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.10599 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.917193 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.90686 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.23936 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.824798 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.744735 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1894 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.26053 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.963976 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.17499 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.16328 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.999496 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.54473 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.970624 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.739894 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.15914 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.00751 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.03186 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.733219 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.71859 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.23952 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.08865 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.41243 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32557 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.859119 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.14866 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.651761 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.26075 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.769326 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.32319 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.792695 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.813563 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.854499 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.4331 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.666191 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.10459 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.606918 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.729382 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.942564 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.986121 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.721607 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.3801 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.973556 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.729209 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.44046 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.28538 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.91212 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.36833 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.12233 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.20546 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.87146 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.928033 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.679329 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.938561 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.94443 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.948759 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.636223 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.61632 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.29535 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.874486 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.52386 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15313 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.92642 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.07805 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.734018 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11294 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.849685 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.14502 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.833084 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.801234 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.02754 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.32371 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.945301 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.21009 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.728633 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.849442 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.03765 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.04965 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.907283 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.14411 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.812657 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.784903 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.2518 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.34451 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.914404 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.07622 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.17042 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.997351 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.51918 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.11019 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.772382 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.05066 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.963779 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.801801 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.760936 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.389 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.09917 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.08537 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.39993 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24143 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.895778 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.06786 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.927111 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.18182 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.805373 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.77554 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.98745 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.30211 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.781589 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.09864 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.509762 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.537797 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.741688 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.66485 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.03432 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.787066 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.535368 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.70816 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.31776 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.911354 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.88505 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.754681 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.17309 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.66972 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.928186 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.03883 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.1067 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.618184 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.807602 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.06468 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.819175 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.39616 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.831032 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.43418 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.76165 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.645045 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.842969 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.06088 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.996165 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.20775 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.82961 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.06897 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.795269 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.16097 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.811653 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.815548 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.93786 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.0964 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.874748 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.942182 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.73943 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.889956 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.19619 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.868771 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.646012 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.04588 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.784235 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.819385 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.2172 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07694 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.984082 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.03339 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.00626 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.11697 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.29671 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.895813 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.782985 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.05572 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00249 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.936598 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.842924 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.24551 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.08985 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.13438 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.29124 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18702 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.917475 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.11933 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.822795 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.17745 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.854737 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.3632 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.934572 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.00406 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.939128 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.15775 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.68647 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.835279 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.733385 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.836201 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.969491 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.95797 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.683014 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.34014 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.15716 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.819278 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.0835 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.984579 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.927923 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.23604 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.99469 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.16954 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.29424 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.720074 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.711845 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02833 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.891866 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11285 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.711221 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.36663 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.27865 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.808476 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.08603 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22299 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.968198 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.14641 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.826824 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.10617 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.738443 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.98859 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.893777 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.38782 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.876362 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.05858 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.449647 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.56277 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.549691 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.731561 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.2319 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.972683 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.720454 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 2.04534 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.2754 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.907688 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.911867 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.880817 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.19033 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.881 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.0826 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.738887 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1656 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.675004 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.736899 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.16272 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.859947 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.42097 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.603313 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.79537 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.67713 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.581048 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.824633 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14353 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.918844 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.11639 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.769273 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.16684 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.895061 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.19813 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.797951 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.820679 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.92226 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.2406 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.846555 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.16718 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.754022 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.879784 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.949556 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.02761 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.831953 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.946141 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.829305 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.899823 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.17992 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13333 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.943221 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.11975 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.09517 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.02946 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.43157 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.977447 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.781782 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.03661 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.949539 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.867059 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.7539 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.16816 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.12533 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.00581 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.27142 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13504 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.923858 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00871 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.81017 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11879 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.883692 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.26632 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.944354 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06743 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.940058 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.14762 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.868035 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.990201 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.855515 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.933785 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.06594 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.961271 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.84923 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.20148 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.97144 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.913626 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.17079 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.08018 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.999431 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.16487 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.00595 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.06157 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.28021 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.937281 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.767797 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07815 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.957473 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.939991 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.891521 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.19983 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16475 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.929551 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.10481 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.332998954950636 z_k1_s0000
                                 - 0.622998679657034 z_k1_s0001
                                 - 0.785997563656361 z_k1_s0002
                                 - 0.454997143126158 z_k1_s0003
                                 - 0.236999400386043 z_k1_s0004
                                 - 0.260999078477842 z_k1_s0005
                                 - 0.485998429268636 z_k1_s0006
                                 - 1.0299947822048 z_k1_s0007
                                 - 0.643000930405024 z_k1_s0010
                                 - 0.283999871914314 z_k1_s0011
                                 - 0.512999791445204 z_k1_s0012
                                 - 1.11300037626349 z_k1_s0013
                                 - 0.340999623797295 z_k1_s0014
                                 - 0.572997825428654 z_k1_s0015
                                 - 0.285999671413871 z_k1_s0016
                                 - 0.29599988361068 z_k1_s0017
                                 - 0.193999904111755 z_k1_s0020
                                 - 0.52699876856221 z_k1_s0021
                                 - 0.644999632809693 z_k1_s0022
                                 - 0.638000058845925 z_k1_s0023
                                 - 0.191000131398717 z_k1_s0024
                                 - 0.485001324779602 z_k1_s0025
                                 - 0.183999931634863 z_k1_s0026
                                 - 0.265000052887731 z_k1_s0027
                                 - 0.701999647828777 z_k1_s0030
                                 - 0.769997299088182 z_k1_s0031
                                 - 0.40699916003097 z_k1_s0032
                                 - 0.576000370268246 z_k1_s0033
                                 - 0.222999284566943 z_k1_s0034
                                 - 0.773997765866753 z_k1_s0035
                                 - 0.63599708797051 z_k1_s0036
                                 - 1.05399958843781 z_k1_s0037
                                 - 0.934998775261784 z_k1_s0040
                                 - 0.951000304422231 z_k1_s0041
                                 - 1.02299986564005 z_k1_s0042
                                 - 0.970003035300786 z_k1_s0043
                                 - 0.565000379447601 z_k1_s0044
                                 - 0.613000574926429 z_k1_s0045
                                 - 1.68600192519099 z_k1_s0046
                                 - 0.577000062317099 z_k1_s0047
                                 - 0.514001197549119 z_k1_s0050
                                 - 0.205000172907175 z_k1_s0051
                                 - 0.232000230961309 z_k1_s0052
                                 - 0.187000217337795 z_k1_s0053
                                 - 0.381001096319591 z_k1_s0054
                                 - 0.266000202459054 z_k1_s0055
                                 - 0.25699935177391 z_k1_s0056
                                 - 0.452000325276278 z_k1_s0057
                                 - 0.312999941256623 z_k1_s0060
                                 - 0.314999410772238 z_k1_s0061
                                 - 0.293000218096663 z_k1_s0062
                                 - 0.192999926295589 z_k1_s0063
                                 - 0.150000102501356 z_k1_s0064
                                 - 0.0810001357987933 z_k1_s0065
                                 - 0.218000837246493 z_k1_s0066
                                 - 0.113000028054984 z_k1_s0067
                                 - 0.330999743585594 z_k1_s0070
                                 - 0.432000303616775 z_k1_s0071
                                 - 0.532999426449481 z_k1_s0072
                                 - 0.403001231386865 z_k1_s0073
                                 - 0.648000455425162 z_k1_s0074
                                 - 0.783000101180881 z_k1_s0075
                                 - 0.248999665993542 z_k1_s0076
                                 - 0.213999930830765 z_k1_s0077
                                 - 0.0879992631852453 z_k1_s0100
                                 - 0.0669995637789256 z_k1_s0101
                                 - 0.187998870256717 z_k1_s0102
                                 - 0.0669995637789256 z_k1_s0103
                                 - 0.0569996782713265 z_k1_s0104
                                 - 0.0539996674993179 z_k1_s0105
                                 - 0.0859997072011948 z_k1_s0106
                                 - 0.14499901665612 z_k1_s0107
                                 - 0.127999384532638 z_k1_s0110
                                 - 0.152998717593792 z_k1_s0111
                                 - 0.227999148566422 z_k1_s0112
                                 - 0.273999142697505 z_k1_s0113
                                 - 0.0879995599787641 z_k1_s0114
                                 - 0.192998947445861 z_k1_s0115
                                 - 0.223998855382418 z_k1_s0116
                                 - 0.0629996865200175 z_k1_s0117
                                 - 0.130999590794305 z_k1_s0120
                                 - 0.0869998614961948 z_k1_s0121
                                 - 0.0719996053329866 z_k1_s0122
                                 - 0.0779996926183595 z_k1_s0123
                                 - 0.0489996881865932 z_k1_s0124
                                 - 0.103999084160799 z_k1_s0125
                                 - 0.0549996995724141 z_k1_s0126
                                 - 0.182999132874705 z_k1_s0127
                                 - 0.0569997620920468 z_k1_s0130
                                 - 0.0559997352940771 z_k1_s0131
                                 - 0.0589997477882852 z_k1_s0132
                                 - 0.121999466969047 z_k1_s0133
                                 - 0.200999411218222 z_k1_s0134
                                 - 0.0639997459787318 z_k1_s0135
                                 - 0.0569997620920468 z_k1_s0136
                                 - 0.062999719180762 z_k1_s0137
                                 - 0.0389998853065876 z_k1_s0140
                                 - 0.0899995495541909 z_k1_s0141
                                 - 0.162999157359333 z_k1_s0142
                                 - 0.10699949130324 z_k1_s0143
                                 - 0.107999298465486 z_k1_s0144
                                 - 0.0709998324811063 z_k1_s0145
                                 - 0.172999643974942 z_k1_s0146
                                 - 0.053999810232286 z_k1_s0147
                                 - 0.0869999024179351 z_k1_s0150
                                 - 0.0570001012568296 z_k1_s0151
                                 - 0.0719998945872191 z_k1_s0152
                                 - 0.136000356998927 z_k1_s0153
                                 - 0.0420000934261136 z_k1_s0154
                                 - 0.128000209822327 z_k1_s0155
                                 - 0.149000506785765 z_k1_s0156
                                 - 0.0440000944702091 z_k1_s0157
                                 - 0.0749999793107068 z_k1_s0160
                                 - 0.0349999306117816 z_k1_s0161
                                 - 0.0299999469243715 z_k1_s0162
                                 - 0.0409999379166204 z_k1_s0163
                                 - 0.0460001008036752 z_k1_s0164
                                 - 0.135999538377589 z_k1_s0165
                                 - 0.0389999354816741 z_k1_s0166
                                 - 0.0460001008036752 z_k1_s0167
                                 - 0.264998302955456 z_k1_s0170
                                 - 0.1049994910419 z_k1_s0171
                                 - 0.185999236736381 z_k1_s0172
                                 - 0.0929994909688352 z_k1_s0173
                                 - 0.132999830944 z_k1_s0174
                                 - 0.160999661248679 z_k1_s0175
                                 - 0.121999406212506 z_k1_s0176
                                 - 0.208998642346098 z_k1_s0177
                                 - 0.379000682579663 z_k1_s1000
                                 - 0.568000650103732 z_k1_s1001
                                 - 0.298000696497918 z_k1_s1002
                                 - 0.796998794903587 z_k1_s1003
                                 - 0.174000002145954 z_k1_s1004
                                 - 0.26299991208272 z_k1_s1005
                                 - 0.17299983141416 z_k1_s1006
                                 - 0.231999810594695 z_k1_s1007
                                 - 1.57399983611262 z_k1_s1010
                                 - 1.74100252352506 z_k1_s1011
                                 - 0.479999191174763 z_k1_s1012
                                 - 2.34199933039157 z_k1_s1013
                                 - 0.582999621232583 z_k1_s1014
                                 - 0.643999062651015 z_k1_s1015
                                 - 0.735999351100799 z_k1_s1016
                                 - 1.75499661157685 z_k1_s1017
                                 - 0.484998100447387 z_k1_s1020
                                 - 0.535000379422644 z_k1_s1021
                                 - 0.997998041958862 z_k1_s1022
                                 - 0.697998489055911 z_k1_s1023
                                 - 0.399998462474027 z_k1_s1024
                                 - 0.360998755945787 z_k1_s1025
                                 - 0.384998720178023 z_k1_s1026
                                 - 0.844997563930927 z_k1_s1027
                                 - 0.606000441084684 z_k1_s1030
                                 - 0.20299993868955 z_k1_s1031
                                 - 0.331998514984734 z_k1_s1032
                                 - 0.281999788057482 z_k1_s1033
                                 - 0.570000022881871 z_k1_s1034
                                 - 0.196999774722472 z_k1_s1035
                                 - 0.321998713039318 z_k1_s1036
                                 - 0.316000528671041 z_k1_s1037
                                 - 0.410999981079681 z_k1_s1040
                                 - 0.24199984223201 z_k1_s1041
                                 - 0.183999830239274 z_k1_s1042
                                 - 0.137999928479388 z_k1_s1043
                                 - 0.33500047333686 z_k1_s1044
                                 - 0.398998531648458 z_k1_s1045
                                 - 0.374000161994431 z_k1_s1046
                                 - 0.148999880726347 z_k1_s1047
                                 - 0.444998085425182 z_k1_s1050
                                 - 0.251999781032262 z_k1_s1051
                                 - 0.27199964299523 z_k1_s1052
                                 - 0.251999781032262 z_k1_s1053
                                 - 0.270999881246908 z_k1_s1054
                                 - 0.430998115951154 z_k1_s1055
                                 - 0.825998198967672 z_k1_s1056
                                 - 0.555999731967844 z_k1_s1057
                                 - 0.813000778718203 z_k1_s1060
                                 - 0.329999794937663 z_k1_s1061
                                 - 0.683001289863653 z_k1_s1062
                                 - 0.357999806229323 z_k1_s1063
                                 - 0.217999749771026 z_k1_s1064
                                 - 0.287999778000174 z_k1_s1065
                                 - 0.788999416469209 z_k1_s1066
                                 - 0.465998835355044 z_k1_s1067
                                 - 0.745000493370458 z_k1_s1070
                                 - 0.726001073775045 z_k1_s1071
                                 - 2.03299734677056 z_k1_s1072
                                 - 0.630000499763023 z_k1_s1073
                                 - 0.554000771379255 z_k1_s1074
                                 - 2.42400450004324 z_k1_s1075
                                 - 1.75700556212117 z_k1_s1076
                                 - 1.38100442432523 z_k1_s1077
                                 - 0.514999724409142 z_k1_s1100
                                 - 0.139000164691104 z_k1_s1101
                                 - 0.591000223984293 z_k1_s1102
                                 - 0.625999149603921 z_k1_s1103
                                 - 0.127999999810454 z_k1_s1104
                                 - 0.31500024438051 z_k1_s1105
                                 - 0.583999799260119 z_k1_s1106
                                 - 0.300000019543259 z_k1_s1107
                                 - 0.109000200059031 z_k1_s1110
                                 - 0.124000013732097 z_k1_s1111
                                 - 0.166000100117237 z_k1_s1112
                                 - 0.297999676641328 z_k1_s1113
                                 - 0.485000732101966 z_k1_s1114
                                 - 0.183000159213625 z_k1_s1115
                                 - 0.24400034741829 z_k1_s1116
                                 - 0.418000213027706 z_k1_s1117
                                 - 0.356000643164091 z_k1_s1120
                                 - 0.261001179118485 z_k1_s1121
                                 - 0.25800121878978 z_k1_s1122
                                 - 0.77500415764793 z_k1_s1123
                                 - 0.237001187787436 z_k1_s1124
                                 - 0.29400136013706 z_k1_s1125
                                 - 0.600001326783287 z_k1_s1126
                                 - 0.306001510153291 z_k1_s1127
                                 - 0.516001418878072 z_k1_s1130
                                 - 0.623003265251727 z_k1_s1131
                                 - 0.733001103997262 z_k1_s1132
                                 - 0.414001015215751 z_k1_s1133
                                 - 1.24100614719536 z_k1_s1134
                                 - 0.525001579131012 z_k1_s1135
                                 - 0.410001238270901 z_k1_s1136
                                 - 0.83500044825634 z_k1_s1137
                                 - 0.320000366001752 z_k1_s1140
                                 - 0.125000100528768 z_k1_s1141
                                 - 0.211000661023446 z_k1_s1142
                                 - 0.259000489573485 z_k1_s1143
                                 - 0.121000190266339 z_k1_s1144
                                 - 0.324999687954228 z_k1_s1145
                                 - 0.0740000752066459 z_k1_s1146
                                 - 0.0740000752066459 z_k1_s1147
                                 - 0.0890001640543334 z_k1_s1150
                                 - 0.135000274824043 z_k1_s1151
                                 - 0.134000332729176 z_k1_s1152
                                 - 0.15200044643921 z_k1_s1153
                                 - 0.0960002211193735 z_k1_s1154
                                 - 0.127000738065105 z_k1_s1155
                                 - 0.122999813683213 z_k1_s1156
                                 - 0.30000112449895 z_k1_s1157
                                 - 0.523999839468144 z_k1_s1160
                                 - 0.306000304093282 z_k1_s1161
                                 - 0.245999480731258 z_k1_s1162
                                 - 0.431999783897551 z_k1_s1163
                                 - 0.14099982279453 z_k1_s1164
                                 - 0.30199827424601 z_k1_s1165
                                 - 0.392999879601108 z_k1_s1166
                                 - 0.39899880987937 z_k1_s1167
                                 - 0.259999735928794 z_k1_s1170
                                 - 0.206000286928648 z_k1_s1171
                                 - 0.308999140391802 z_k1_s1172
                                 - 0.554001322711525 z_k1_s1173
                                 - 0.201000242391511 z_k1_s1174
                                 - 0.395000938431498 z_k1_s1175
                                 - 0.336999905800238 z_k1_s1176
                                 - 0.318000768560053 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.05623 w_k1_ADM_s000
                                 + 0.756668 w_k1_AXP_s000
                                 + 0.819726 w_k1_DIS_s000
                                 + 0.801982 w_k1_FDX_s000
                                 + 1.18404 w_k1_IBM_s000
                                 + 0.766551 w_k1_INTC_s000
                                 + 1.00161 w_k1_KO_s000 + 0.86315 w_k1_MAT_s000
                                 + 0.731477 w_k1_MSFT_s000
                                 + 0.886669 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.978394 w_k1_ADM_s000
                                 + 1.06911 w_k1_AXP_s000
                                 + 1.38712 w_k1_DIS_s000
                                 + 1.13371 w_k1_FDX_s000
                                 + 0.949952 w_k1_IBM_s000
                                 + 1.08959 w_k1_INTC_s000
                                 + 1.14663 w_k1_KO_s000
                                 + 0.896455 w_k1_MAT_s000
                                 + 1.08123 w_k1_MSFT_s000
                                 + 1.11498 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.05034 w_k1_ADM_s000
                                 + 0.867675 w_k1_AXP_s000
                                 + 0.885336 w_k1_DIS_s000
                                 + 0.920307 w_k1_FDX_s000
                                 + 0.686022 w_k1_IBM_s000
                                 + 0.971195 w_k1_INTC_s000
                                 + 0.892958 w_k1_KO_s000
                                 + 0.69205 w_k1_MAT_s000
                                 + 0.897777 w_k1_MSFT_s000
                                 + 0.806502 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.27929 w_k1_ADM_s000
                                 + 1.38533 w_k1_AXP_s000
                                 + 1.36072 w_k1_DIS_s000 + 1.2192 w_k1_FDX_s000
                                 + 1.0912 w_k1_IBM_s000
                                 + 1.07686 w_k1_INTC_s000
                                 + 1.34982 w_k1_KO_s000 + 1.14445 w_k1_MAT_s000
                                 + 1.1287 w_k1_MSFT_s000 + 1.25455 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.02524 w_k1_ADM_s000
                                 + 1.07664 w_k1_AXP_s000 + 1.0559 w_k1_DIS_s000
                                 + 1.02029 w_k1_FDX_s000
                                 + 1.17595 w_k1_IBM_s000
                                 + 0.919147 w_k1_INTC_s000
                                 + 1.21344 w_k1_KO_s000 + 1.13327 w_k1_MAT_s000
                                 + 0.967255 w_k1_MSFT_s000
                                 + 1.09653 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.29718 w_k1_ADM_s000
                                 + 1.81431 w_k1_AXP_s000
                                 + 2.27824 w_k1_DIS_s000
                                 + 1.67234 w_k1_FDX_s000
                                 + 1.27524 w_k1_IBM_s000
                                 + 1.28398 w_k1_INTC_s000
                                 + 1.58551 w_k1_KO_s000 + 1.26476 w_k1_MAT_s000
                                 + 1.29434 w_k1_MSFT_s000
                                 + 1.38728 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.16681 w_k1_ADM_s000
                                 + 1.10275 w_k1_AXP_s000
                                 + 1.17209 w_k1_DIS_s000
                                 + 0.980333 w_k1_FDX_s000
                                 + 1.66909 w_k1_IBM_s000
                                 + 0.826238 w_k1_INTC_s000
                                 + 1.34031 w_k1_KO_s000 + 1.59002 w_k1_MAT_s000
                                 + 0.813124 w_k1_MSFT_s000
                                 + 1.17581 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.681304 w_k1_ADM_s000
                                 + 0.671488 w_k1_AXP_s000
                                 + 0.711996 w_k1_DIS_s000
                                 + 0.76305 w_k1_FDX_s000
                                 + 0.901375 w_k1_IBM_s000
                                 + 0.787214 w_k1_INTC_s000
                                 + 0.813426 w_k1_KO_s000
                                 + 0.837924 w_k1_MAT_s000
                                 + 0.741874 w_k1_MSFT_s000
                                 + 0.777132 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.04113 w_k1_ADM_s001
                                 + 0.789048 w_k1_AXP_s001
                                 + 0.845239 w_k1_DIS_s001
                                 + 0.89991 w_k1_FDX_s001
                                 + 0.825984 w_k1_IBM_s001
                                 + 0.846974 w_k1_INTC_s001
                                 + 0.869079 w_k1_KO_s001
                                 + 1.07485 w_k1_MAT_s001
                                 + 0.922771 w_k1_MSFT_s001
                                 + 0.942588 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.797688 w_k1_ADM_s001
                                 + 0.582866 w_k1_AXP_s001
                                 + 0.581338 w_k1_DIS_s001
                                 + 0.699046 w_k1_FDX_s001
                                 + 0.755626 w_k1_IBM_s001
                                 + 0.705761 w_k1_INTC_s001
                                 + 0.71399 w_k1_KO_s001
                                 + 0.839066 w_k1_MAT_s001
                                 + 0.77937 w_k1_MSFT_s001
                                 + 0.770455 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.13487 w_k1_ADM_s001
                                 + 1.04512 w_k1_AXP_s001
                                 + 1.18641 w_k1_DIS_s001
                                 + 1.13836 w_k1_FDX_s001
                                 + 0.887549 w_k1_IBM_s001
                                 + 0.978922 w_k1_INTC_s001
                                 + 0.981388 w_k1_KO_s001
                                 + 1.05769 w_k1_MAT_s001
                                 + 1.17554 w_k1_MSFT_s001
                                 + 1.07934 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17005 w_k1_ADM_s001
                                 + 0.953665 w_k1_AXP_s001
                                 + 0.930176 w_k1_DIS_s001
                                 + 0.906596 w_k1_FDX_s001
                                 + 1.11396 w_k1_IBM_s001
                                 + 0.876074 w_k1_INTC_s001
                                 + 1.01993 w_k1_KO_s001 + 1.27566 w_k1_MAT_s001
                                 + 0.951293 w_k1_MSFT_s001
                                 + 0.958544 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.20901 w_k1_ADM_s001
                                 + 1.21042 w_k1_AXP_s001
                                 + 1.75691 w_k1_DIS_s001
                                 + 1.39204 w_k1_FDX_s001
                                 + 1.16732 w_k1_IBM_s001
                                 + 1.28843 w_k1_INTC_s001
                                 + 1.45872 w_k1_KO_s001 + 1.28552 w_k1_MAT_s001
                                 + 1.42826 w_k1_MSFT_s001
                                 + 1.22242 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.13967 w_k1_ADM_s001
                                 + 1.21521 w_k1_AXP_s001
                                 + 1.37331 w_k1_DIS_s001
                                 + 1.17268 w_k1_FDX_s001
                                 + 1.28914 w_k1_IBM_s001
                                 + 1.04747 w_k1_INTC_s001 + 1.2413 w_k1_KO_s001
                                 + 1.56976 w_k1_MAT_s001
                                 + 1.14515 w_k1_MSFT_s001
                                 + 1.16132 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.12705 w_k1_ADM_s001
                                 + 0.722151 w_k1_AXP_s001
                                 + 0.868075 w_k1_DIS_s001
                                 + 0.9451 w_k1_FDX_s001
                                 + 0.775306 w_k1_IBM_s001
                                 + 0.909401 w_k1_INTC_s001
                                 + 0.821646 w_k1_KO_s001
                                 + 0.581082 w_k1_MAT_s001
                                 + 1.05156 w_k1_MSFT_s001
                                 + 0.956852 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.967451 w_k1_ADM_s001
                                 + 0.877469 w_k1_AXP_s001
                                 + 0.886846 w_k1_DIS_s001
                                 + 0.813815 w_k1_FDX_s001
                                 + 1.52995 w_k1_IBM_s001
                                 + 0.773565 w_k1_INTC_s001
                                 + 1.05233 w_k1_KO_s001 + 1.23985 w_k1_MAT_s001
                                 + 0.82638 w_k1_MSFT_s001
                                 + 0.992939 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.26633 w_k1_ADM_s002
                                 + 1.89389 w_k1_AXP_s002
                                 + 1.57033 w_k1_DIS_s002
                                 + 1.46575 w_k1_FDX_s002
                                 + 1.42304 w_k1_IBM_s002
                                 + 1.83113 w_k1_INTC_s002
                                 + 1.53778 w_k1_KO_s002 + 1.40757 w_k1_MAT_s002
                                 + 1.37745 w_k1_MSFT_s002
                                 + 1.22864 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.957113 w_k1_ADM_s002
                                 + 1.20796 w_k1_AXP_s002
                                 + 0.995946 w_k1_DIS_s002
                                 + 1.03302 w_k1_FDX_s002
                                 + 1.37788 w_k1_IBM_s002
                                 + 1.09852 w_k1_INTC_s002
                                 + 1.11498 w_k1_KO_s002 + 1.54682 w_k1_MAT_s002
                                 + 1.02697 w_k1_MSFT_s002
                                 + 0.860532 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.00443 w_k1_ADM_s002
                                 + 1.08938 w_k1_AXP_s002
                                 + 0.939482 w_k1_DIS_s002
                                 + 1.09588 w_k1_FDX_s002
                                 + 0.760114 w_k1_IBM_s002
                                 + 1.04118 w_k1_INTC_s002
                                 + 0.802697 w_k1_KO_s002
                                 + 0.898677 w_k1_MAT_s002
                                 + 0.96266 w_k1_MSFT_s002
                                 + 0.87578 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.05233 w_k1_ADM_s002
                                 + 1.06833 w_k1_AXP_s002
                                 + 1.29117 w_k1_DIS_s002
                                 + 1.13279 w_k1_FDX_s002
                                 + 1.13959 w_k1_IBM_s002
                                 + 1.00727 w_k1_INTC_s002
                                 + 1.15607 w_k1_KO_s002 + 1.11261 w_k1_MAT_s002
                                 + 1.15033 w_k1_MSFT_s002
                                 + 1.04042 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.857179 w_k1_ADM_s002
                                 + 0.685346 w_k1_AXP_s002
                                 + 0.571426 w_k1_DIS_s002
                                 + 0.709534 w_k1_FDX_s002
                                 + 0.94355 w_k1_IBM_s002
                                 + 0.789289 w_k1_INTC_s002
                                 + 0.826915 w_k1_KO_s002
                                 + 0.600392 w_k1_MAT_s002
                                 + 0.672195 w_k1_MSFT_s002
                                 + 0.786434 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.941654 w_k1_ADM_s002
                                 + 0.789541 w_k1_AXP_s002
                                 + 0.80057 w_k1_DIS_s002
                                 + 0.887419 w_k1_FDX_s002
                                 + 1.03234 w_k1_IBM_s002
                                 + 0.80106 w_k1_INTC_s002
                                 + 0.806146 w_k1_KO_s002
                                 + 0.784333 w_k1_MAT_s002
                                 + 0.873288 w_k1_MSFT_s002
                                 + 0.819739 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.783942 w_k1_ADM_s002
                                 + 0.661744 w_k1_AXP_s002
                                 + 0.744161 w_k1_DIS_s002
                                 + 0.803438 w_k1_FDX_s002
                                 + 0.668948 w_k1_IBM_s002
                                 + 0.903308 w_k1_INTC_s002
                                 + 0.746077 w_k1_KO_s002
                                 + 0.604905 w_k1_MAT_s002
                                 + 0.874374 w_k1_MSFT_s002
                                 + 0.73627 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.39127 w_k1_ADM_s002
                                 + 1.29762 w_k1_AXP_s002
                                 + 1.24404 w_k1_DIS_s002
                                 + 1.42343 w_k1_FDX_s002
                                 + 0.767816 w_k1_IBM_s002
                                 + 1.33814 w_k1_INTC_s002
                                 + 1.01266 w_k1_KO_s002
                                 + 0.782621 w_k1_MAT_s002
                                 + 1.29589 w_k1_MSFT_s002
                                 + 0.96538 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.82115 w_k1_ADM_s003
                                 + 1.35101 w_k1_AXP_s003
                                 + 1.01397 w_k1_DIS_s003 + 1.0376 w_k1_FDX_s003
                                 + 1.10997 w_k1_IBM_s003
                                 + 1.05474 w_k1_INTC_s003
                                 + 0.834968 w_k1_KO_s003
                                 + 0.709274 w_k1_MAT_s003
                                 + 0.923928 w_k1_MSFT_s003
                                 + 1.10581 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.961371 w_k1_ADM_s003
                                 + 1.2171 w_k1_AXP_s003 + 1.51216 w_k1_DIS_s003
                                 + 1.18198 w_k1_FDX_s003
                                 + 1.30713 w_k1_IBM_s003
                                 + 1.16612 w_k1_INTC_s003
                                 + 1.22727 w_k1_KO_s003 + 1.39101 w_k1_MAT_s003
                                 + 1.15092 w_k1_MSFT_s003
                                 + 1.38596 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.854774 w_k1_ADM_s003
                                 + 1.03209 w_k1_AXP_s003
                                 + 1.08415 w_k1_DIS_s003
                                 + 0.94079 w_k1_FDX_s003
                                 + 1.33251 w_k1_IBM_s003
                                 + 0.898221 w_k1_INTC_s003
                                 + 1.07058 w_k1_KO_s003 + 1.35012 w_k1_MAT_s003
                                 + 1.0226 w_k1_MSFT_s003
                                 + 0.972366 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.956347 w_k1_ADM_s003
                                 + 1.19489 w_k1_AXP_s003
                                 + 1.18336 w_k1_DIS_s003
                                 + 1.23416 w_k1_FDX_s003
                                 + 0.829216 w_k1_IBM_s003
                                 + 1.26868 w_k1_INTC_s003
                                 + 0.958327 w_k1_KO_s003
                                 + 0.702602 w_k1_MAT_s003
                                 + 1.13722 w_k1_MSFT_s003
                                 + 1.12298 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.2646 w_k1_ADM_s003
                                 + 1.55165 w_k1_AXP_s003
                                 + 1.61839 w_k1_DIS_s003 + 1.4809 w_k1_FDX_s003
                                 + 1.02006 w_k1_IBM_s003
                                 + 1.08158 w_k1_INTC_s003
                                 + 1.08525 w_k1_KO_s003 + 1.07435 w_k1_MAT_s003
                                 + 1.25483 w_k1_MSFT_s003
                                 + 1.19063 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.676239 w_k1_ADM_s003
                                 + 0.890723 w_k1_AXP_s003
                                 + 0.636469 w_k1_DIS_s003
                                 + 0.762584 w_k1_FDX_s003
                                 + 0.710196 w_k1_IBM_s003
                                 + 0.822837 w_k1_INTC_s003
                                 + 0.602297 w_k1_KO_s003
                                 + 0.626007 w_k1_MAT_s003
                                 + 0.765528 w_k1_MSFT_s003
                                 + 0.903059 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.847192 w_k1_ADM_s003
                                 + 0.838824 w_k1_AXP_s003
                                 + 1.03533 w_k1_DIS_s003
                                 + 0.975034 w_k1_FDX_s003
                                 + 0.826988 w_k1_IBM_s003
                                 + 0.915403 w_k1_INTC_s003
                                 + 0.877453 w_k1_KO_s003 + 0.7773 w_k1_MAT_s003
                                 + 1.15658 w_k1_MSFT_s003
                                 + 1.06807 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.741601 w_k1_ADM_s003
                                 + 0.911861 w_k1_AXP_s003
                                 + 0.795523 w_k1_DIS_s003
                                 + 0.836566 w_k1_FDX_s003
                                 + 1.04798 w_k1_IBM_s003
                                 + 0.98974 w_k1_INTC_s003
                                 + 0.846234 w_k1_KO_s003
                                 + 1.00253 w_k1_MAT_s003
                                 + 0.947139 w_k1_MSFT_s003
                                 + 0.931998 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.780852 w_k1_ADM_s004
                                 + 0.736632 w_k1_AXP_s004
                                 + 0.502794 w_k1_DIS_s004
                                 + 0.701264 w_k1_FDX_s004
                                 + 0.876653 w_k1_IBM_s004
                                 + 0.754555 w_k1_INTC_s004
                                 + 0.810231 w_k1_KO_s004
                                 + 1.02818 w_k1_MAT_s004
                                 + 0.749314 w_k1_MSFT_s004
                                 + 0.885014 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01843 w_k1_ADM_s004
                                 + 1.6757 w_k1_AXP_s004 + 1.64173 w_k1_DIS_s004
                                 + 1.38311 w_k1_FDX_s004
                                 + 1.21247 w_k1_IBM_s004
                                 + 1.35575 w_k1_INTC_s004
                                 + 1.42539 w_k1_KO_s004 + 1.79298 w_k1_MAT_s004
                                 + 1.22083 w_k1_MSFT_s004
                                 + 1.29177 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.11399 w_k1_ADM_s004
                                 + 0.977368 w_k1_AXP_s004
                                 + 0.940362 w_k1_DIS_s004
                                 + 1.03103 w_k1_FDX_s004
                                 + 0.758884 w_k1_IBM_s004
                                 + 1.00451 w_k1_INTC_s004
                                 + 0.979317 w_k1_KO_s004
                                 + 0.586956 w_k1_MAT_s004
                                 + 0.965836 w_k1_MSFT_s004
                                 + 0.965235 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.38257 w_k1_ADM_s004
                                 + 1.22109 w_k1_AXP_s004
                                 + 1.13316 w_k1_DIS_s004 + 1.1997 w_k1_FDX_s004
                                 + 0.935659 w_k1_IBM_s004
                                 + 1.39414 w_k1_INTC_s004
                                 + 1.11593 w_k1_KO_s004 + 1.11827 w_k1_MAT_s004
                                 + 1.11078 w_k1_MSFT_s004
                                 + 1.21406 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.766271 w_k1_ADM_s004
                                 + 0.794795 w_k1_AXP_s004
                                 + 0.697192 w_k1_DIS_s004
                                 + 0.894764 w_k1_FDX_s004
                                 + 0.693038 w_k1_IBM_s004
                                 + 0.869532 w_k1_INTC_s004
                                 + 0.796431 w_k1_KO_s004
                                 + 0.801394 w_k1_MAT_s004
                                 + 0.789845 w_k1_MSFT_s004
                                 + 0.944768 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.973057 w_k1_ADM_s004
                                 + 1.04841 w_k1_AXP_s004
                                 + 0.695281 w_k1_DIS_s004
                                 + 0.860506 w_k1_FDX_s004
                                 + 0.977819 w_k1_IBM_s004
                                 + 0.952916 w_k1_INTC_s004
                                 + 0.953154 w_k1_KO_s004
                                 + 1.38423 w_k1_MAT_s004
                                 + 0.788438 w_k1_MSFT_s004
                                 + 0.923139 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.05902 w_k1_ADM_s004
                                 + 1.07235 w_k1_AXP_s004 + 1.0612 w_k1_DIS_s004
                                 + 0.906119 w_k1_FDX_s004
                                 + 1.31315 w_k1_IBM_s004
                                 + 1.03527 w_k1_INTC_s004
                                 + 1.29964 w_k1_KO_s004 + 1.53857 w_k1_MAT_s004
                                 + 0.960595 w_k1_MSFT_s004
                                 + 1.09393 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.843885 w_k1_ADM_s004
                                 + 0.670597 w_k1_AXP_s004
                                 + 0.936638 w_k1_DIS_s004
                                 + 0.823236 w_k1_FDX_s004
                                 + 0.85995 w_k1_IBM_s004
                                 + 0.926836 w_k1_INTC_s004
                                 + 1.01538 w_k1_KO_s004
                                 + 0.786479 w_k1_MAT_s004
                                 + 0.885105 w_k1_MSFT_s004
                                 + 1.02908 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.13496 w_k1_ADM_s005
                                 + 0.948879 w_k1_AXP_s005
                                 + 1.02268 w_k1_DIS_s005
                                 + 0.959245 w_k1_FDX_s005
                                 + 0.92703 w_k1_IBM_s005
                                 + 0.935491 w_k1_INTC_s005
                                 + 1.00183 w_k1_KO_s005 + 1.5131 w_k1_MAT_s005
                                 + 0.987281 w_k1_MSFT_s005
                                 + 1.10127 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.687936 w_k1_ADM_s005
                                 + 0.645017 w_k1_AXP_s005
                                 + 0.880058 w_k1_DIS_s005
                                 + 0.792919 w_k1_FDX_s005
                                 + 0.860913 w_k1_IBM_s005
                                 + 0.817169 w_k1_INTC_s005
                                 + 0.907922 w_k1_KO_s005
                                 + 0.947088 w_k1_MAT_s005
                                 + 0.829191 w_k1_MSFT_s005
                                 + 0.871466 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.12134 w_k1_ADM_s005
                                 + 1.80106 w_k1_AXP_s005
                                 + 1.56104 w_k1_DIS_s005
                                 + 1.39438 w_k1_FDX_s005
                                 + 1.23026 w_k1_IBM_s005
                                 + 1.27713 w_k1_INTC_s005
                                 + 1.22318 w_k1_KO_s005 + 2.00343 w_k1_MAT_s005
                                 + 1.29039 w_k1_MSFT_s005
                                 + 1.07481 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.938124 w_k1_ADM_s005
                                 + 1.04101 w_k1_AXP_s005
                                 + 0.840169 w_k1_DIS_s005
                                 + 0.940517 w_k1_FDX_s005
                                 + 0.800542 w_k1_IBM_s005
                                 + 0.840937 w_k1_INTC_s005
                                 + 0.805943 w_k1_KO_s005
                                 + 1.00233 w_k1_MAT_s005
                                 + 0.876856 w_k1_MSFT_s005
                                 + 0.861203 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.967618 w_k1_ADM_s005
                                 + 1.01515 w_k1_AXP_s005
                                 + 0.81039 w_k1_DIS_s005
                                 + 0.866311 w_k1_FDX_s005
                                 + 1.25056 w_k1_IBM_s005
                                 + 0.915856 w_k1_INTC_s005
                                 + 0.81118 w_k1_KO_s005 + 1.18338 w_k1_MAT_s005
                                 + 0.787275 w_k1_MSFT_s005
                                 + 0.819667 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.968922 w_k1_ADM_s005
                                 + 0.714727 w_k1_AXP_s005
                                 + 0.681982 w_k1_DIS_s005
                                 + 0.730926 w_k1_FDX_s005
                                 + 0.648125 w_k1_IBM_s005
                                 + 0.715693 w_k1_INTC_s005
                                 + 0.667097 w_k1_KO_s005
                                 + 0.640022 w_k1_MAT_s005
                                 + 0.803062 w_k1_MSFT_s005
                                 + 0.752696 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.23836 w_k1_ADM_s005
                                 + 1.35328 w_k1_AXP_s005
                                 + 1.02904 w_k1_DIS_s005
                                 + 1.22261 w_k1_FDX_s005
                                 + 0.781407 w_k1_IBM_s005
                                 + 1.08655 w_k1_INTC_s005
                                 + 0.84344 w_k1_KO_s005 + 1.10285 w_k1_MAT_s005
                                 + 1.11878 w_k1_MSFT_s005
                                 + 1.03497 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.29762 w_k1_ADM_s005
                                 + 1.21642 w_k1_AXP_s005
                                 + 1.35028 w_k1_DIS_s005 + 1.127 w_k1_FDX_s005
                                 + 1.24318 w_k1_IBM_s005
                                 + 0.97764 w_k1_INTC_s005
                                 + 1.10985 w_k1_KO_s005 + 1.42907 w_k1_MAT_s005
                                 + 1.24566 w_k1_MSFT_s005 + 1.1717 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.869855 w_k1_ADM_s006
                                 + 0.729235 w_k1_AXP_s006
                                 + 0.70359 w_k1_DIS_s006
                                 + 0.739938 w_k1_FDX_s006
                                 + 0.85064 w_k1_IBM_s006
                                 + 0.85848 w_k1_INTC_s006
                                 + 0.852436 w_k1_KO_s006
                                 + 0.892686 w_k1_MAT_s006
                                 + 0.799067 w_k1_MSFT_s006
                                 + 0.81358 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.13977 w_k1_ADM_s006
                                 + 1.18475 w_k1_AXP_s006
                                 + 1.49325 w_k1_DIS_s006
                                 + 1.17952 w_k1_FDX_s006
                                 + 1.44186 w_k1_IBM_s006
                                 + 1.11207 w_k1_INTC_s006
                                 + 1.40409 w_k1_KO_s006 + 1.41814 w_k1_MAT_s006
                                 + 1.28054 w_k1_MSFT_s006
                                 + 1.13069 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.96592 w_k1_ADM_s006
                                 + 0.909586 w_k1_AXP_s006
                                 + 0.881167 w_k1_DIS_s006
                                 + 0.941743 w_k1_FDX_s006
                                 + 0.82261 w_k1_IBM_s006
                                 + 1.08412 w_k1_INTC_s006
                                 + 0.828258 w_k1_KO_s006
                                 + 0.602106 w_k1_MAT_s006
                                 + 0.893544 w_k1_MSFT_s006
                                 + 0.865936 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.1326 w_k1_ADM_s006
                                 + 1.47339 w_k1_AXP_s006
                                 + 1.16383 w_k1_DIS_s006
                                 + 1.22691 w_k1_FDX_s006
                                 + 1.16375 w_k1_IBM_s006
                                 + 1.17171 w_k1_INTC_s006
                                 + 0.93113 w_k1_KO_s006 + 1.2711 w_k1_MAT_s006
                                 + 0.938921 w_k1_MSFT_s006
                                 + 1.01888 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.00837 w_k1_ADM_s006
                                 + 0.778527 w_k1_AXP_s006
                                 + 1.08887 w_k1_DIS_s006
                                 + 0.918482 w_k1_FDX_s006
                                 + 1.29538 w_k1_IBM_s006
                                 + 0.992129 w_k1_INTC_s006
                                 + 1.21863 w_k1_KO_s006 + 1.10246 w_k1_MAT_s006
                                 + 0.922457 w_k1_MSFT_s006
                                 + 1.13924 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.886645 w_k1_ADM_s006
                                 + 0.800646 w_k1_AXP_s006
                                 + 0.916623 w_k1_DIS_s006
                                 + 0.812741 w_k1_FDX_s006
                                 + 1.15063 w_k1_IBM_s006
                                 + 0.873029 w_k1_INTC_s006
                                 + 1.01176 w_k1_KO_s006
                                 + 0.693997 w_k1_MAT_s006
                                 + 0.933471 w_k1_MSFT_s006
                                 + 0.881757 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.25697 w_k1_ADM_s006
                                 + 1.14114 w_k1_AXP_s006
                                 + 1.29461 w_k1_DIS_s006 + 1.2572 w_k1_FDX_s006
                                 + 0.818685 w_k1_IBM_s006
                                 + 1.15107 w_k1_INTC_s006
                                 + 1.02085 w_k1_KO_s006 + 0.79669 w_k1_MAT_s006
                                 + 0.953661 w_k1_MSFT_s006
                                 + 0.932012 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.06832 w_k1_ADM_s006
                                 + 1.10261 w_k1_AXP_s006
                                 + 0.87309 w_k1_DIS_s006
                                 + 0.884371 w_k1_FDX_s006
                                 + 1.44286 w_k1_IBM_s006
                                 + 1.00143 w_k1_INTC_s006
                                 + 1.04125 w_k1_KO_s006 + 1.31817 w_k1_MAT_s006
                                 + 0.899498 w_k1_MSFT_s006
                                 + 0.995152 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.970944 w_k1_ADM_s007
                                 + 1.33149 w_k1_AXP_s007
                                 + 1.65303 w_k1_DIS_s007
                                 + 1.35578 w_k1_FDX_s007
                                 + 1.20711 w_k1_IBM_s007
                                 + 1.06231 w_k1_INTC_s007
                                 + 1.26645 w_k1_KO_s007 + 1.86312 w_k1_MAT_s007
                                 + 1.31997 w_k1_MSFT_s007 + 1.364 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.990305 w_k1_ADM_s007
                                 + 0.872309 w_k1_AXP_s007
                                 + 0.854255 w_k1_DIS_s007
                                 + 0.921291 w_k1_FDX_s007
                                 + 0.715462 w_k1_IBM_s007
                                 + 0.872993 w_k1_INTC_s007
                                 + 0.80861 w_k1_KO_s007
                                 + 0.722694 w_k1_MAT_s007
                                 + 0.790752 w_k1_MSFT_s007
                                 + 0.905941 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.936308 w_k1_ADM_s007
                                 + 1.05963 w_k1_AXP_s007
                                 + 0.993476 w_k1_DIS_s007
                                 + 1.01814 w_k1_FDX_s007
                                 + 0.875804 w_k1_IBM_s007
                                 + 1.08309 w_k1_INTC_s007
                                 + 0.900242 w_k1_KO_s007
                                 + 1.19587 w_k1_MAT_s007
                                 + 1.05664 w_k1_MSFT_s007
                                 + 1.02534 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.766366 w_k1_ADM_s007
                                 + 1.01736 w_k1_AXP_s007
                                 + 0.661864 w_k1_DIS_s007
                                 + 0.734999 w_k1_FDX_s007
                                 + 1.18048 w_k1_IBM_s007
                                 + 0.763789 w_k1_INTC_s007
                                 + 0.887758 w_k1_KO_s007
                                 + 1.41125 w_k1_MAT_s007
                                 + 0.693708 w_k1_MSFT_s007
                                 + 0.919978 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.924793 w_k1_ADM_s007
                                 + 1.43879 w_k1_AXP_s007
                                 + 1.17087 w_k1_DIS_s007
                                 + 1.08189 w_k1_FDX_s007
                                 + 1.53427 w_k1_IBM_s007
                                 + 1.08798 w_k1_INTC_s007
                                 + 1.12813 w_k1_KO_s007 + 2.03238 w_k1_MAT_s007
                                 + 0.990192 w_k1_MSFT_s007
                                 + 1.24818 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.21346 w_k1_ADM_s007
                                 + 1.32966 w_k1_AXP_s007
                                 + 1.16468 w_k1_DIS_s007
                                 + 1.13523 w_k1_FDX_s007
                                 + 0.93418 w_k1_IBM_s007
                                 + 1.06557 w_k1_INTC_s007
                                 + 0.995947 w_k1_KO_s007 + 1.0256 w_k1_MAT_s007
                                 + 1.00839 w_k1_MSFT_s007
                                 + 0.970279 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.90212 w_k1_ADM_s007
                                 + 0.801705 w_k1_AXP_s007
                                 + 0.690991 w_k1_DIS_s007
                                 + 0.80648 w_k1_FDX_s007
                                 + 0.868278 w_k1_IBM_s007
                                 + 0.702531 w_k1_INTC_s007
                                 + 0.760331 w_k1_KO_s007
                                 + 1.01677 w_k1_MAT_s007
                                 + 0.820442 w_k1_MSFT_s007
                                 + 0.807663 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.737972 w_k1_ADM_s007
                                 + 0.6964 w_k1_AXP_s007 + 0.48772 w_k1_DIS_s007
                                 + 0.737315 w_k1_FDX_s007
                                 + 0.641119 w_k1_IBM_s007
                                 + 0.655425 w_k1_INTC_s007
                                 + 0.616161 w_k1_KO_s007
                                 + 0.613318 w_k1_MAT_s007
                                 + 0.701256 w_k1_MSFT_s007
                                 + 0.699937 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.929425 w_k1_ADM_s010
                                 + 1.14902 w_k1_AXP_s010
                                 + 1.10014 w_k1_DIS_s010
                                 + 1.01226 w_k1_FDX_s010
                                 + 1.11304 w_k1_IBM_s010
                                 + 0.989852 w_k1_INTC_s010
                                 + 0.931815 w_k1_KO_s010
                                 + 1.05079 w_k1_MAT_s010
                                 + 1.10241 w_k1_MSFT_s010
                                 + 0.911534 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.965155 w_k1_ADM_s010
                                 + 1.09055 w_k1_AXP_s010
                                 + 1.09942 w_k1_DIS_s010
                                 + 1.22261 w_k1_FDX_s010
                                 + 0.908799 w_k1_IBM_s010
                                 + 1.05259 w_k1_INTC_s010
                                 + 0.901248 w_k1_KO_s010
                                 + 0.809282 w_k1_MAT_s010
                                 + 1.06896 w_k1_MSFT_s010
                                 + 0.903458 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.768469 w_k1_ADM_s010
                                 + 0.655842 w_k1_AXP_s010
                                 + 0.686389 w_k1_DIS_s010
                                 + 0.745059 w_k1_FDX_s010
                                 + 1.02252 w_k1_IBM_s010
                                 + 0.898947 w_k1_INTC_s010
                                 + 0.72887 w_k1_KO_s010
                                 + 0.816108 w_k1_MAT_s010
                                 + 0.807025 w_k1_MSFT_s010
                                 + 0.796487 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.42747 w_k1_ADM_s010
                                 + 1.34877 w_k1_AXP_s010
                                 + 1.48754 w_k1_DIS_s010 + 1.4176 w_k1_FDX_s010
                                 + 0.871545 w_k1_IBM_s010
                                 + 1.31823 w_k1_INTC_s010
                                 + 0.937839 w_k1_KO_s010
                                 + 0.463041 w_k1_MAT_s010
                                 + 1.28235 w_k1_MSFT_s010
                                 + 0.919695 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.861548 w_k1_ADM_s010
                                 + 1.01223 w_k1_AXP_s010 + 1.1093 w_k1_DIS_s010
                                 + 1.01208 w_k1_FDX_s010
                                 + 1.55954 w_k1_IBM_s010
                                 + 0.757514 w_k1_INTC_s010
                                 + 1.0739 w_k1_KO_s010 + 1.99587 w_k1_MAT_s010
                                 + 0.922066 w_k1_MSFT_s010
                                 + 1.02621 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.14625 w_k1_ADM_s010
                                 + 1.65224 w_k1_AXP_s010
                                 + 1.60622 w_k1_DIS_s010 + 1.5194 w_k1_FDX_s010
                                 + 1.17747 w_k1_IBM_s010
                                 + 1.26952 w_k1_INTC_s010
                                 + 1.08976 w_k1_KO_s010 + 1.36561 w_k1_MAT_s010
                                 + 1.33246 w_k1_MSFT_s010
                                 + 1.07211 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.832245 w_k1_ADM_s010
                                 + 1.12044 w_k1_AXP_s010
                                 + 1.04831 w_k1_DIS_s010
                                 + 1.06246 w_k1_FDX_s010
                                 + 1.32572 w_k1_IBM_s010
                                 + 1.24865 w_k1_INTC_s010
                                 + 1.04422 w_k1_KO_s010 + 1.2465 w_k1_MAT_s010
                                 + 1.13103 w_k1_MSFT_s010
                                 + 1.00431 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.993263 w_k1_ADM_s010
                                 + 0.763856 w_k1_AXP_s010
                                 + 0.972542 w_k1_DIS_s010
                                 + 0.970719 w_k1_FDX_s010
                                 + 0.714635 w_k1_IBM_s010
                                 + 0.904788 w_k1_INTC_s010
                                 + 0.809474 w_k1_KO_s010
                                 + 0.531432 w_k1_MAT_s010
                                 + 1.07822 w_k1_MSFT_s010
                                 + 0.820478 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.93616 w_k1_ADM_s011
                                 + 1.34567 w_k1_AXP_s011
                                 + 1.55302 w_k1_DIS_s011 + 1.4074 w_k1_FDX_s011
                                 + 1.12295 w_k1_IBM_s011
                                 + 0.917734 w_k1_INTC_s011
                                 + 1.32859 w_k1_KO_s011 + 1.10824 w_k1_MAT_s011
                                 + 1.34964 w_k1_MSFT_s011
                                 + 1.22321 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.31827 w_k1_ADM_s011
                                 + 1.14755 w_k1_AXP_s011
                                 + 1.26083 w_k1_DIS_s011
                                 + 1.14638 w_k1_FDX_s011
                                 + 1.55076 w_k1_IBM_s011
                                 + 1.05429 w_k1_INTC_s011
                                 + 1.18952 w_k1_KO_s011 + 1.0542 w_k1_MAT_s011
                                 + 1.25787 w_k1_MSFT_s011
                                 + 1.05149 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.28853 w_k1_ADM_s011
                                 + 1.30014 w_k1_AXP_s011
                                 + 1.18295 w_k1_DIS_s011
                                 + 1.05242 w_k1_FDX_s011
                                 + 0.975039 w_k1_IBM_s011
                                 + 1.01076 w_k1_INTC_s011
                                 + 1.22564 w_k1_KO_s011 + 1.02748 w_k1_MAT_s011
                                 + 1.23695 w_k1_MSFT_s011
                                 + 1.02105 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.977418 w_k1_ADM_s011
                                 + 0.7677 w_k1_AXP_s011
                                 + 0.959396 w_k1_DIS_s011
                                 + 0.899591 w_k1_FDX_s011
                                 + 1.12924 w_k1_IBM_s011
                                 + 0.788491 w_k1_INTC_s011
                                 + 1.07074 w_k1_KO_s011 + 1.00665 w_k1_MAT_s011
                                 + 0.92939 w_k1_MSFT_s011
                                 + 0.866607 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.64035 w_k1_ADM_s011
                                 + 1.56139 w_k1_AXP_s011
                                 + 1.18523 w_k1_DIS_s011 + 1.1991 w_k1_FDX_s011
                                 + 1.02866 w_k1_IBM_s011
                                 + 1.27226 w_k1_INTC_s011
                                 + 1.07654 w_k1_KO_s011
                                 + 0.854116 w_k1_MAT_s011
                                 + 1.22431 w_k1_MSFT_s011
                                 + 1.03219 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.31885 w_k1_ADM_s011
                                 + 1.01844 w_k1_AXP_s011
                                 + 0.826861 w_k1_DIS_s011
                                 + 0.967203 w_k1_FDX_s011
                                 + 0.79168 w_k1_IBM_s011
                                 + 0.746046 w_k1_INTC_s011
                                 + 0.850093 w_k1_KO_s011
                                 + 0.815933 w_k1_MAT_s011
                                 + 0.941354 w_k1_MSFT_s011
                                 + 0.875997 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.0192 w_k1_ADM_s011
                                 + 0.700235 w_k1_AXP_s011
                                 + 0.723047 w_k1_DIS_s011
                                 + 0.811572 w_k1_FDX_s011
                                 + 0.730083 w_k1_IBM_s011
                                 + 0.736554 w_k1_INTC_s011
                                 + 0.672695 w_k1_KO_s011
                                 + 0.501825 w_k1_MAT_s011
                                 + 0.778655 w_k1_MSFT_s011
                                 + 0.782339 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.711207 w_k1_ADM_s011
                                 + 0.846617 w_k1_AXP_s011
                                 + 0.71191 w_k1_DIS_s011
                                 + 0.676992 w_k1_FDX_s011
                                 + 1.10887 w_k1_IBM_s011
                                 + 0.633593 w_k1_INTC_s011
                                 + 0.857396 w_k1_KO_s011
                                 + 1.08853 w_k1_MAT_s011
                                 + 0.671221 w_k1_MSFT_s011
                                 + 0.812322 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.888826 w_k1_ADM_s012
                                 + 0.977245 w_k1_AXP_s012
                                 + 1.07958 w_k1_DIS_s012
                                 + 0.980537 w_k1_FDX_s012
                                 + 1.04477 w_k1_IBM_s012
                                 + 1.18444 w_k1_INTC_s012
                                 + 0.952543 w_k1_KO_s012
                                 + 0.665771 w_k1_MAT_s012
                                 + 0.917001 w_k1_MSFT_s012
                                 + 1.04624 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.973004 w_k1_ADM_s012
                                 + 0.679873 w_k1_AXP_s012
                                 + 0.979041 w_k1_DIS_s012
                                 + 0.931845 w_k1_FDX_s012
                                 + 0.849731 w_k1_IBM_s012
                                 + 1.42445 w_k1_INTC_s012
                                 + 0.964789 w_k1_KO_s012
                                 + 0.490038 w_k1_MAT_s012
                                 + 1.03737 w_k1_MSFT_s012
                                 + 0.972675 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.73766 w_k1_ADM_s012
                                 + 0.878929 w_k1_AXP_s012
                                 + 0.804875 w_k1_DIS_s012
                                 + 0.868893 w_k1_FDX_s012
                                 + 1.04552 w_k1_IBM_s012
                                 + 1.06162 w_k1_INTC_s012
                                 + 0.890452 w_k1_KO_s012
                                 + 0.843385 w_k1_MAT_s012
                                 + 0.924226 w_k1_MSFT_s012
                                 + 1.04615 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.909083 w_k1_ADM_s012
                                 + 0.894001 w_k1_AXP_s012
                                 + 0.721622 w_k1_DIS_s012
                                 + 0.837904 w_k1_FDX_s012
                                 + 0.627572 w_k1_IBM_s012
                                 + 0.972701 w_k1_INTC_s012
                                 + 0.657654 w_k1_KO_s012
                                 + 0.639135 w_k1_MAT_s012
                                 + 0.877943 w_k1_MSFT_s012
                                 + 0.79802 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.858555 w_k1_ADM_s012
                                 + 0.550985 w_k1_AXP_s012
                                 + 0.609563 w_k1_DIS_s012
                                 + 0.682245 w_k1_FDX_s012
                                 + 0.890336 w_k1_IBM_s012
                                 + 0.940689 w_k1_INTC_s012
                                 + 0.839905 w_k1_KO_s012
                                 + 0.608424 w_k1_MAT_s012
                                 + 0.831884 w_k1_MSFT_s012
                                 + 0.926442 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02944 w_k1_ADM_s012
                                 + 1.41591 w_k1_AXP_s012
                                 + 1.08582 w_k1_DIS_s012 + 1.2374 w_k1_FDX_s012
                                 + 0.806636 w_k1_IBM_s012
                                 + 1.47678 w_k1_INTC_s012
                                 + 0.984475 w_k1_KO_s012
                                 + 0.841862 w_k1_MAT_s012
                                 + 1.05398 w_k1_MSFT_s012 + 1.1437 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.962469 w_k1_ADM_s012
                                 + 1.0102 w_k1_AXP_s012 + 0.79659 w_k1_DIS_s012
                                 + 0.907511 w_k1_FDX_s012
                                 + 0.807942 w_k1_IBM_s012
                                 + 1.03306 w_k1_INTC_s012
                                 + 0.677823 w_k1_KO_s012
                                 + 0.874469 w_k1_MAT_s012
                                 + 0.892773 w_k1_MSFT_s012
                                 + 1.01385 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.19111 w_k1_ADM_s012
                                 + 1.34853 w_k1_AXP_s012
                                 + 1.27774 w_k1_DIS_s012
                                 + 1.06699 w_k1_FDX_s012
                                 + 1.44099 w_k1_IBM_s012
                                 + 1.27628 w_k1_INTC_s012
                                 + 1.24482 w_k1_KO_s012 + 1.19556 w_k1_MAT_s012
                                 + 1.21359 w_k1_MSFT_s012
                                 + 1.20443 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.859435 w_k1_ADM_s013
                                 + 0.9281 w_k1_AXP_s013
                                 + 0.723288 w_k1_DIS_s013
                                 + 0.839986 w_k1_FDX_s013
                                 + 1.26657 w_k1_IBM_s013
                                 + 1.36761 w_k1_INTC_s013
                                 + 1.06048 w_k1_KO_s013 + 1.12406 w_k1_MAT_s013
                                 + 1.0271 w_k1_MSFT_s013 + 1.07543 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.739032 w_k1_ADM_s013
                                 + 0.515936 w_k1_AXP_s013
                                 + 0.515419 w_k1_DIS_s013
                                 + 0.614833 w_k1_FDX_s013
                                 + 1.00771 w_k1_IBM_s013
                                 + 1.0251 w_k1_INTC_s013
                                 + 0.826888 w_k1_KO_s013
                                 + 0.703613 w_k1_MAT_s013
                                 + 0.709523 w_k1_MSFT_s013
                                 + 0.814666 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.00466 w_k1_ADM_s013
                                 + 1.13988 w_k1_AXP_s013
                                 + 1.18745 w_k1_DIS_s013
                                 + 1.17568 w_k1_FDX_s013
                                 + 1.29812 w_k1_IBM_s013
                                 + 1.32878 w_k1_INTC_s013
                                 + 1.21908 w_k1_KO_s013 + 1.1693 w_k1_MAT_s013
                                 + 1.03843 w_k1_MSFT_s013 + 1.1904 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.12157 w_k1_ADM_s013
                                 + 0.869543 w_k1_AXP_s013
                                 + 0.957235 w_k1_DIS_s013
                                 + 1.05367 w_k1_FDX_s013
                                 + 0.899154 w_k1_IBM_s013
                                 + 1.32579 w_k1_INTC_s013
                                 + 0.872549 w_k1_KO_s013
                                 + 0.731772 w_k1_MAT_s013
                                 + 1.10315 w_k1_MSFT_s013
                                 + 1.04635 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.664369 w_k1_ADM_s013
                                 + 0.766746 w_k1_AXP_s013
                                 + 0.713894 w_k1_DIS_s013
                                 + 0.874248 w_k1_FDX_s013
                                 + 0.743828 w_k1_IBM_s013
                                 + 1.32309 w_k1_INTC_s013
                                 + 0.749827 w_k1_KO_s013
                                 + 0.543702 w_k1_MAT_s013
                                 + 0.893156 w_k1_MSFT_s013
                                 + 0.838861 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.956743 w_k1_ADM_s013
                                 + 1.15157 w_k1_AXP_s013
                                 + 0.841124 w_k1_DIS_s013
                                 + 0.940788 w_k1_FDX_s013
                                 + 1.4007 w_k1_IBM_s013
                                 + 1.18396 w_k1_INTC_s013
                                 + 1.16115 w_k1_KO_s013 + 1.424 w_k1_MAT_s013
                                 + 1.02371 w_k1_MSFT_s013
                                 + 0.967138 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.35429 w_k1_ADM_s013
                                 + 1.77335 w_k1_AXP_s013
                                 + 1.40517 w_k1_DIS_s013
                                 + 1.31533 w_k1_FDX_s013
                                 + 1.36361 w_k1_IBM_s013
                                 + 1.45769 w_k1_INTC_s013
                                 + 1.26923 w_k1_KO_s013 + 1.70546 w_k1_MAT_s013
                                 + 1.37952 w_k1_MSFT_s013
                                 + 1.18115 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.29519 w_k1_ADM_s013
                                 + 1.17093 w_k1_AXP_s013
                                 + 1.69655 w_k1_DIS_s013
                                 + 1.53759 w_k1_FDX_s013
                                 + 0.94767 w_k1_IBM_s013
                                 + 1.35467 w_k1_INTC_s013
                                 + 1.21363 w_k1_KO_s013
                                 + 0.932697 w_k1_MAT_s013
                                 + 1.22134 w_k1_MSFT_s013
                                 + 1.12725 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.634235 w_k1_ADM_s014
                                 + 0.542498 w_k1_AXP_s014
                                 + 0.539148 w_k1_DIS_s014
                                 + 0.614116 w_k1_FDX_s014
                                 + 0.969256 w_k1_IBM_s014
                                 + 0.813635 w_k1_INTC_s014
                                 + 0.838075 w_k1_KO_s014
                                 + 0.759897 w_k1_MAT_s014
                                 + 0.890252 w_k1_MSFT_s014
                                 + 0.633282 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.762248 w_k1_ADM_s014
                                 + 0.949011 w_k1_AXP_s014
                                 + 0.74609 w_k1_DIS_s014
                                 + 0.799228 w_k1_FDX_s014
                                 + 1.00336 w_k1_IBM_s014
                                 + 1.01941 w_k1_INTC_s014
                                 + 0.950461 w_k1_KO_s014
                                 + 0.987028 w_k1_MAT_s014
                                 + 1.21709 w_k1_MSFT_s014
                                 + 0.867441 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.841716 w_k1_ADM_s014
                                 + 0.886135 w_k1_AXP_s014
                                 + 1.01152 w_k1_DIS_s014
                                 + 0.95816 w_k1_FDX_s014
                                 + 0.724015 w_k1_IBM_s014
                                 + 0.987152 w_k1_INTC_s014
                                 + 0.920367 w_k1_KO_s014
                                 + 0.683709 w_k1_MAT_s014
                                 + 1.26749 w_k1_MSFT_s014
                                 + 0.964025 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.09417 w_k1_ADM_s014
                                 + 1.67827 w_k1_AXP_s014 + 1.5025 w_k1_DIS_s014
                                 + 1.21445 w_k1_FDX_s014
                                 + 1.59082 w_k1_IBM_s014
                                 + 1.15289 w_k1_INTC_s014
                                 + 1.38446 w_k1_KO_s014 + 1.59532 w_k1_MAT_s014
                                 + 1.30601 w_k1_MSFT_s014
                                 + 1.14064 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.26958 w_k1_ADM_s014
                                 + 1.18942 w_k1_AXP_s014
                                 + 1.27607 w_k1_DIS_s014
                                 + 1.16057 w_k1_FDX_s014
                                 + 1.05155 w_k1_IBM_s014
                                 + 1.49122 w_k1_INTC_s014
                                 + 1.14443 w_k1_KO_s014
                                 + 0.594418 w_k1_MAT_s014
                                 + 1.36941 w_k1_MSFT_s014
                                 + 1.12163 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.01864 w_k1_ADM_s014
                                 + 1.16233 w_k1_AXP_s014
                                 + 0.999127 w_k1_DIS_s014
                                 + 0.966367 w_k1_FDX_s014
                                 + 0.874773 w_k1_IBM_s014
                                 + 1.15758 w_k1_INTC_s014
                                 + 1.01559 w_k1_KO_s014
                                 + 0.885893 w_k1_MAT_s014
                                 + 1.16068 w_k1_MSFT_s014
                                 + 0.918514 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.817114 w_k1_ADM_s014
                                 + 0.986333 w_k1_AXP_s014
                                 + 0.986957 w_k1_DIS_s014
                                 + 0.905172 w_k1_FDX_s014
                                 + 1.27698 w_k1_IBM_s014
                                 + 1.10587 w_k1_INTC_s014
                                 + 1.05179 w_k1_KO_s014 + 1.20916 w_k1_MAT_s014
                                 + 1.12749 w_k1_MSFT_s014
                                 + 1.11775 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.733227 w_k1_ADM_s014
                                 + 0.877013 w_k1_AXP_s014
                                 + 0.630344 w_k1_DIS_s014
                                 + 0.783761 w_k1_FDX_s014
                                 + 0.74625 w_k1_IBM_s014
                                 + 0.883746 w_k1_INTC_s014
                                 + 0.762037 w_k1_KO_s014
                                 + 0.554293 w_k1_MAT_s014
                                 + 0.962268 w_k1_MSFT_s014
                                 + 0.753907 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.07478 w_k1_ADM_s015
                                 + 0.996253 w_k1_AXP_s015
                                 + 0.875138 w_k1_DIS_s015
                                 + 1.02464 w_k1_FDX_s015
                                 + 0.870341 w_k1_IBM_s015
                                 + 1.28512 w_k1_INTC_s015
                                 + 0.886756 w_k1_KO_s015
                                 + 0.959144 w_k1_MAT_s015
                                 + 1.13911 w_k1_MSFT_s015
                                 + 1.00771 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.785937 w_k1_ADM_s015
                                 + 0.694521 w_k1_AXP_s015
                                 + 0.577338 w_k1_DIS_s015
                                 + 0.726479 w_k1_FDX_s015
                                 + 1.09964 w_k1_IBM_s015
                                 + 0.93205 w_k1_INTC_s015
                                 + 0.839386 w_k1_KO_s015
                                 + 1.16588 w_k1_MAT_s015
                                 + 0.874302 w_k1_MSFT_s015
                                 + 0.855801 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.12634 w_k1_ADM_s015
                                 + 1.10743 w_k1_AXP_s015
                                 + 1.15706 w_k1_DIS_s015
                                 + 1.03192 w_k1_FDX_s015
                                 + 1.01891 w_k1_IBM_s015
                                 + 1.31385 w_k1_INTC_s015
                                 + 1.04227 w_k1_KO_s015
                                 + 0.632427 w_k1_MAT_s015
                                 + 1.12089 w_k1_MSFT_s015
                                 + 0.994714 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.837608 w_k1_ADM_s015
                                 + 1.26954 w_k1_AXP_s015
                                 + 0.860123 w_k1_DIS_s015
                                 + 1.07923 w_k1_FDX_s015
                                 + 1.15105 w_k1_IBM_s015
                                 + 1.09315 w_k1_INTC_s015
                                 + 0.89578 w_k1_KO_s015 + 1.41189 w_k1_MAT_s015
                                 + 0.998757 w_k1_MSFT_s015
                                 + 1.06941 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.66333 w_k1_ADM_s015
                                 + 0.68485 w_k1_AXP_s015
                                 + 0.523617 w_k1_DIS_s015
                                 + 0.594314 w_k1_FDX_s015
                                 + 0.752097 w_k1_IBM_s015
                                 + 0.70756 w_k1_INTC_s015
                                 + 0.667703 w_k1_KO_s015
                                 + 0.650281 w_k1_MAT_s015
                                 + 0.649165 w_k1_MSFT_s015
                                 + 0.751653 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.07796 w_k1_ADM_s015
                                 + 1.08973 w_k1_AXP_s015
                                 + 0.764168 w_k1_DIS_s015
                                 + 0.961426 w_k1_FDX_s015
                                 + 0.702574 w_k1_IBM_s015
                                 + 0.992654 w_k1_INTC_s015
                                 + 0.68659 w_k1_KO_s015
                                 + 0.744947 w_k1_MAT_s015
                                 + 0.920957 w_k1_MSFT_s015
                                 + 0.872491 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.13477 w_k1_ADM_s015
                                 + 1.72953 w_k1_AXP_s015
                                 + 1.20339 w_k1_DIS_s015 + 1.2924 w_k1_FDX_s015
                                 + 1.27702 w_k1_IBM_s015
                                 + 1.65348 w_k1_INTC_s015
                                 + 1.35736 w_k1_KO_s015 + 1.58886 w_k1_MAT_s015
                                 + 1.35976 w_k1_MSFT_s015
                                 + 1.29583 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06666 w_k1_ADM_s015
                                 + 0.989268 w_k1_AXP_s015
                                 + 0.581661 w_k1_DIS_s015
                                 + 0.798439 w_k1_FDX_s015
                                 + 0.911917 w_k1_IBM_s015
                                 + 0.955204 w_k1_INTC_s015
                                 + 0.795777 w_k1_KO_s015
                                 + 1.18414 w_k1_MAT_s015
                                 + 0.844617 w_k1_MSFT_s015
                                 + 0.917371 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04783 w_k1_ADM_s016
                                 + 1.21548 w_k1_AXP_s016
                                 + 1.04289 w_k1_DIS_s016
                                 + 1.13257 w_k1_FDX_s016
                                 + 0.796782 w_k1_IBM_s016
                                 + 1.35769 w_k1_INTC_s016
                                 + 1.03538 w_k1_KO_s016
                                 + 0.767606 w_k1_MAT_s016
                                 + 0.912174 w_k1_MSFT_s016
                                 + 1.0878 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.695214 w_k1_ADM_s016
                                 + 0.685386 w_k1_AXP_s016
                                 + 0.669636 w_k1_DIS_s016
                                 + 0.633374 w_k1_FDX_s016
                                 + 1.0596 w_k1_IBM_s016
                                 + 1.00379 w_k1_INTC_s016
                                 + 0.906059 w_k1_KO_s016
                                 + 0.760365 w_k1_MAT_s016
                                 + 0.731076 w_k1_MSFT_s016
                                 + 0.777483 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.37043 w_k1_ADM_s016
                                 + 1.09028 w_k1_AXP_s016
                                 + 1.15593 w_k1_DIS_s016 + 1.1598 w_k1_FDX_s016
                                 + 1.00141 w_k1_IBM_s016
                                 + 1.21204 w_k1_INTC_s016
                                 + 1.19229 w_k1_KO_s016 + 0.81967 w_k1_MAT_s016
                                 + 1.06007 w_k1_MSFT_s016 + 1.1121 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.30094 w_k1_ADM_s016
                                 + 1.1345 w_k1_AXP_s016 + 1.53426 w_k1_DIS_s016
                                 + 1.27429 w_k1_FDX_s016
                                 + 1.29837 w_k1_IBM_s016
                                 + 1.43812 w_k1_INTC_s016
                                 + 1.48584 w_k1_KO_s016 + 1.09116 w_k1_MAT_s016
                                 + 1.19125 w_k1_MSFT_s016
                                 + 1.15831 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.882059 w_k1_ADM_s016
                                 + 1.07241 w_k1_AXP_s016
                                 + 1.40513 w_k1_DIS_s016
                                 + 1.11945 w_k1_FDX_s016
                                 + 1.06756 w_k1_IBM_s016
                                 + 1.01848 w_k1_INTC_s016
                                 + 1.16438 w_k1_KO_s016 + 1.1853 w_k1_MAT_s016
                                 + 0.997062 w_k1_MSFT_s016
                                 + 1.06557 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.642263 w_k1_ADM_s016
                                 + 0.778016 w_k1_AXP_s016
                                 + 0.767751 w_k1_DIS_s016
                                 + 0.852586 w_k1_FDX_s016
                                 + 0.844685 w_k1_IBM_s016
                                 + 1.00515 w_k1_INTC_s016
                                 + 0.844686 w_k1_KO_s016
                                 + 0.774438 w_k1_MAT_s016
                                 + 0.86794 w_k1_MSFT_s016 + 0.832 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.892047 w_k1_ADM_s016
                                 + 1.10356 w_k1_AXP_s016
                                 + 1.11026 w_k1_DIS_s016
                                 + 0.912036 w_k1_FDX_s016
                                 + 1.25031 w_k1_IBM_s016
                                 + 0.931656 w_k1_INTC_s016
                                 + 1.14006 w_k1_KO_s016 + 1.39016 w_k1_MAT_s016
                                 + 0.919459 w_k1_MSFT_s016
                                 + 0.916858 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.09088 w_k1_ADM_s016
                                 + 1.39129 w_k1_AXP_s016
                                 + 1.45897 w_k1_DIS_s016
                                 + 1.05674 w_k1_FDX_s016
                                 + 1.58027 w_k1_IBM_s016
                                 + 1.0787 w_k1_INTC_s016 + 1.55085 w_k1_KO_s016
                                 + 2.41486 w_k1_MAT_s016
                                 + 1.1473 w_k1_MSFT_s016 + 1.21956 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.81075 w_k1_ADM_s017
                                 + 0.793091 w_k1_AXP_s017
                                 + 0.613835 w_k1_DIS_s017
                                 + 0.854216 w_k1_FDX_s017
                                 + 0.579966 w_k1_IBM_s017
                                 + 0.928589 w_k1_INTC_s017
                                 + 0.64968 w_k1_KO_s017
                                 + 0.627235 w_k1_MAT_s017
                                 + 0.873662 w_k1_MSFT_s017
                                 + 0.87475 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.19695 w_k1_ADM_s017
                                 + 0.90557 w_k1_AXP_s017
                                 + 1.06553 w_k1_DIS_s017
                                 + 0.996326 w_k1_FDX_s017
                                 + 1.24923 w_k1_IBM_s017
                                 + 1.01025 w_k1_INTC_s017
                                 + 1.13824 w_k1_KO_s017 + 1.41769 w_k1_MAT_s017
                                 + 1.09974 w_k1_MSFT_s017
                                 + 1.10798 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.21441 w_k1_ADM_s017
                                 + 1.14383 w_k1_AXP_s017
                                 + 1.21407 w_k1_DIS_s017
                                 + 1.31477 w_k1_FDX_s017
                                 + 0.892791 w_k1_IBM_s017
                                 + 0.996894 w_k1_INTC_s017
                                 + 1.0149 w_k1_KO_s017 + 1.17852 w_k1_MAT_s017
                                 + 1.09524 w_k1_MSFT_s017
                                 + 0.971163 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.46719 w_k1_ADM_s017
                                 + 1.5374 w_k1_AXP_s017 + 1.28526 w_k1_DIS_s017
                                 + 1.31295 w_k1_FDX_s017
                                 + 1.15242 w_k1_IBM_s017
                                 + 0.92087 w_k1_INTC_s017
                                 + 1.04123 w_k1_KO_s017 + 1.35663 w_k1_MAT_s017
                                 + 1.00185 w_k1_MSFT_s017
                                 + 1.16415 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.927449 w_k1_ADM_s017
                                 + 0.905539 w_k1_AXP_s017
                                 + 0.958583 w_k1_DIS_s017
                                 + 0.862319 w_k1_FDX_s017
                                 + 1.54686 w_k1_IBM_s017
                                 + 0.914534 w_k1_INTC_s017
                                 + 1.2222 w_k1_KO_s017 + 1.18919 w_k1_MAT_s017
                                 + 0.945932 w_k1_MSFT_s017
                                 + 1.01175 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.963279 w_k1_ADM_s017
                                 + 0.871636 w_k1_AXP_s017
                                 + 1.0037 w_k1_DIS_s017 + 1.01593 w_k1_FDX_s017
                                 + 0.850766 w_k1_IBM_s017
                                 + 0.914982 w_k1_INTC_s017
                                 + 1.05906 w_k1_KO_s017
                                 + 0.921739 w_k1_MAT_s017
                                 + 1.05834 w_k1_MSFT_s017
                                 + 0.881023 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.50667 w_k1_ADM_s017
                                 + 1.57712 w_k1_AXP_s017
                                 + 1.60541 w_k1_DIS_s017
                                 + 1.35722 w_k1_FDX_s017
                                 + 1.54485 w_k1_IBM_s017
                                 + 1.57918 w_k1_INTC_s017 + 1.5854 w_k1_KO_s017
                                 + 1.24974 w_k1_MAT_s017
                                 + 1.52009 w_k1_MSFT_s017
                                 + 1.10677 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.03819 w_k1_ADM_s017
                                 + 1.1345 w_k1_AXP_s017
                                 + 0.764252 w_k1_DIS_s017
                                 + 1.00844 w_k1_FDX_s017
                                 + 1.04686 w_k1_IBM_s017
                                 + 1.16881 w_k1_INTC_s017
                                 + 0.974282 w_k1_KO_s017 + 1.2185 w_k1_MAT_s017
                                 + 0.97101 w_k1_MSFT_s017
                                 + 0.898827 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.22085 w_k1_ADM_s100
                                 + 1.18756 w_k1_AXP_s100
                                 + 1.52991 w_k1_DIS_s100
                                 + 1.28047 w_k1_FDX_s100
                                 + 1.39627 w_k1_IBM_s100
                                 + 1.20258 w_k1_INTC_s100
                                 + 1.29657 w_k1_KO_s100 + 1.50107 w_k1_MAT_s100
                                 + 1.26048 w_k1_MSFT_s100
                                 + 1.22019 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.16362 w_k1_ADM_s100
                                 + 0.892953 w_k1_AXP_s100
                                 + 0.830529 w_k1_DIS_s100
                                 + 0.95986 w_k1_FDX_s100
                                 + 0.995567 w_k1_IBM_s100
                                 + 0.947124 w_k1_INTC_s100
                                 + 0.953334 w_k1_KO_s100
                                 + 0.947627 w_k1_MAT_s100
                                 + 0.893796 w_k1_MSFT_s100
                                 + 1.05986 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.1844 w_k1_ADM_s100
                                 + 1.18386 w_k1_AXP_s100
                                 + 1.03089 w_k1_DIS_s100
                                 + 1.18628 w_k1_FDX_s100
                                 + 0.794121 w_k1_IBM_s100
                                 + 1.05134 w_k1_INTC_s100
                                 + 0.911683 w_k1_KO_s100
                                 + 0.803495 w_k1_MAT_s100
                                 + 1.18121 w_k1_MSFT_s100
                                 + 1.03375 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.963759 w_k1_ADM_s100
                                 + 0.724948 w_k1_AXP_s100
                                 + 0.84389 w_k1_DIS_s100
                                 + 0.876171 w_k1_FDX_s100
                                 + 0.841544 w_k1_IBM_s100
                                 + 0.853731 w_k1_INTC_s100
                                 + 0.880679 w_k1_KO_s100
                                 + 0.728563 w_k1_MAT_s100
                                 + 0.886754 w_k1_MSFT_s100
                                 + 0.902418 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.10593 w_k1_ADM_s100
                                 + 0.812391 w_k1_AXP_s100
                                 + 1.48795 w_k1_DIS_s100
                                 + 1.13976 w_k1_FDX_s100
                                 + 0.864254 w_k1_IBM_s100
                                 + 1.31136 w_k1_INTC_s100
                                 + 1.06378 w_k1_KO_s100 + 0.79952 w_k1_MAT_s100
                                 + 1.25656 w_k1_MSFT_s100
                                 + 0.975413 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.30219 w_k1_ADM_s100
                                 + 1.42868 w_k1_AXP_s100
                                 + 1.26977 w_k1_DIS_s100
                                 + 1.20184 w_k1_FDX_s100
                                 + 1.06068 w_k1_IBM_s100
                                 + 0.963692 w_k1_INTC_s100
                                 + 0.939825 w_k1_KO_s100
                                 + 0.890015 w_k1_MAT_s100
                                 + 1.12159 w_k1_MSFT_s100
                                 + 1.12685 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.912771 w_k1_ADM_s100
                                 + 0.848666 w_k1_AXP_s100
                                 + 1.1133 w_k1_DIS_s100
                                 + 0.949822 w_k1_FDX_s100
                                 + 1.40122 w_k1_IBM_s100
                                 + 1.12882 w_k1_INTC_s100
                                 + 1.12001 w_k1_KO_s100 + 1.41186 w_k1_MAT_s100
                                 + 1.14745 w_k1_MSFT_s100 + 1.0268 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.918248 w_k1_ADM_s100
                                 + 0.592385 w_k1_AXP_s100
                                 + 0.588504 w_k1_DIS_s100
                                 + 0.658459 w_k1_FDX_s100
                                 + 0.934731 w_k1_IBM_s100
                                 + 0.893201 w_k1_INTC_s100
                                 + 0.743159 w_k1_KO_s100
                                 + 0.915595 w_k1_MAT_s100
                                 + 0.804552 w_k1_MSFT_s100
                                 + 0.786352 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01433 w_k1_ADM_s101
                                 + 0.908818 w_k1_AXP_s101
                                 + 0.668877 w_k1_DIS_s101
                                 + 0.839875 w_k1_FDX_s101
                                 + 1.08253 w_k1_IBM_s101
                                 + 0.775771 w_k1_INTC_s101
                                 + 0.922185 w_k1_KO_s101
                                 + 1.14931 w_k1_MAT_s101
                                 + 0.791187 w_k1_MSFT_s101
                                 + 1.0077 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.42868 w_k1_ADM_s101
                                 + 1.34549 w_k1_AXP_s101
                                 + 1.42107 w_k1_DIS_s101
                                 + 1.29289 w_k1_FDX_s101 + 1.1698 w_k1_IBM_s101
                                 + 1.08926 w_k1_INTC_s101
                                 + 1.30068 w_k1_KO_s101 + 1.0573 w_k1_MAT_s101
                                 + 1.1504 w_k1_MSFT_s101 + 1.22389 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.13882 w_k1_ADM_s101
                                 + 1.01762 w_k1_AXP_s101
                                 + 1.26424 w_k1_DIS_s101
                                 + 1.33551 w_k1_FDX_s101
                                 + 0.711009 w_k1_IBM_s101
                                 + 1.03649 w_k1_INTC_s101
                                 + 1.05282 w_k1_KO_s101
                                 + 0.800554 w_k1_MAT_s101
                                 + 1.15447 w_k1_MSFT_s101
                                 + 0.988053 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.03987 w_k1_ADM_s101
                                 + 0.854807 w_k1_AXP_s101
                                 + 0.842076 w_k1_DIS_s101
                                 + 0.944248 w_k1_FDX_s101
                                 + 0.727988 w_k1_IBM_s101
                                 + 0.812663 w_k1_INTC_s101
                                 + 0.884783 w_k1_KO_s101
                                 + 0.587698 w_k1_MAT_s101
                                 + 0.84068 w_k1_MSFT_s101
                                 + 0.918105 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.693153 w_k1_ADM_s101
                                 + 0.568635 w_k1_AXP_s101
                                 + 0.476369 w_k1_DIS_s101
                                 + 0.642764 w_k1_FDX_s101
                                 + 0.752724 w_k1_IBM_s101
                                 + 0.727734 w_k1_INTC_s101
                                 + 0.77229 w_k1_KO_s101
                                 + 0.754068 w_k1_MAT_s101
                                 + 0.713762 w_k1_MSFT_s101
                                 + 0.729596 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.35342 w_k1_ADM_s101
                                 + 1.44683 w_k1_AXP_s101
                                 + 0.939178 w_k1_DIS_s101
                                 + 1.23027 w_k1_FDX_s101
                                 + 0.91477 w_k1_IBM_s101
                                 + 1.21868 w_k1_INTC_s101
                                 + 0.981951 w_k1_KO_s101
                                 + 0.978094 w_k1_MAT_s101
                                 + 0.964381 w_k1_MSFT_s101
                                 + 1.0445 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.03562 w_k1_ADM_s101
                                 + 1.30738 w_k1_AXP_s101
                                 + 1.11055 w_k1_DIS_s101
                                 + 0.975672 w_k1_FDX_s101
                                 + 1.68895 w_k1_IBM_s101
                                 + 1.04271 w_k1_INTC_s101
                                 + 1.33662 w_k1_KO_s101 + 1.96909 w_k1_MAT_s101
                                 + 0.97478 w_k1_MSFT_s101
                                 + 1.14216 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.10042 w_k1_ADM_s101
                                 + 0.809446 w_k1_AXP_s101
                                 + 0.925069 w_k1_DIS_s101
                                 + 0.903267 w_k1_FDX_s101
                                 + 1.18123 w_k1_IBM_s101
                                 + 0.912116 w_k1_INTC_s101
                                 + 1.1113 w_k1_KO_s101 + 1.08367 w_k1_MAT_s101
                                 + 0.998662 w_k1_MSFT_s101
                                 + 1.02031 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.03935 w_k1_ADM_s102
                                 + 1.02365 w_k1_AXP_s102
                                 + 0.699452 w_k1_DIS_s102
                                 + 0.963857 w_k1_FDX_s102
                                 + 0.699249 w_k1_IBM_s102
                                 + 0.79607 w_k1_INTC_s102
                                 + 0.723714 w_k1_KO_s102
                                 + 0.797359 w_k1_MAT_s102
                                 + 0.7632 w_k1_MSFT_s102
                                 + 0.839181 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.39832 w_k1_ADM_s102
                                 + 1.27314 w_k1_AXP_s102
                                 + 1.40454 w_k1_DIS_s102
                                 + 1.36644 w_k1_FDX_s102
                                 + 0.958658 w_k1_IBM_s102
                                 + 1.15031 w_k1_INTC_s102
                                 + 1.22174 w_k1_KO_s102
                                 + 0.669203 w_k1_MAT_s102
                                 + 1.16796 w_k1_MSFT_s102
                                 + 1.23387 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.05994 w_k1_ADM_s102
                                 + 0.937255 w_k1_AXP_s102
                                 + 1.03325 w_k1_DIS_s102
                                 + 1.03387 w_k1_FDX_s102
                                 + 0.818407 w_k1_IBM_s102
                                 + 0.939725 w_k1_INTC_s102
                                 + 0.990754 w_k1_KO_s102
                                 + 0.53222 w_k1_MAT_s102
                                 + 0.966016 w_k1_MSFT_s102
                                 + 1.01506 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.916244 w_k1_ADM_s102
                                 + 0.789341 w_k1_AXP_s102
                                 + 0.606073 w_k1_DIS_s102
                                 + 0.739202 w_k1_FDX_s102
                                 + 0.843998 w_k1_IBM_s102
                                 + 0.893067 w_k1_INTC_s102
                                 + 0.758051 w_k1_KO_s102
                                 + 0.620695 w_k1_MAT_s102
                                 + 0.733841 w_k1_MSFT_s102
                                 + 0.809114 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.5911 w_k1_ADM_s102
                                 + 1.63166 w_k1_AXP_s102
                                 + 1.51586 w_k1_DIS_s102
                                 + 1.42971 w_k1_FDX_s102
                                 + 1.48425 w_k1_IBM_s102
                                 + 1.38667 w_k1_INTC_s102
                                 + 1.61633 w_k1_KO_s102 + 1.55659 w_k1_MAT_s102
                                 + 1.27322 w_k1_MSFT_s102
                                 + 1.35254 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.11515 w_k1_ADM_s102
                                 + 1.07597 w_k1_AXP_s102
                                 + 0.987639 w_k1_DIS_s102
                                 + 1.02549 w_k1_FDX_s102
                                 + 1.64028 w_k1_IBM_s102
                                 + 0.963075 w_k1_INTC_s102
                                 + 1.38052 w_k1_KO_s102 + 1.62734 w_k1_MAT_s102
                                 + 0.99911 w_k1_MSFT_s102
                                 + 1.15588 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.688111 w_k1_ADM_s102
                                 + 0.812378 w_k1_AXP_s102
                                 + 0.647008 w_k1_DIS_s102
                                 + 0.704933 w_k1_FDX_s102
                                 + 1.38687 w_k1_IBM_s102
                                 + 0.886989 w_k1_INTC_s102
                                 + 1.01945 w_k1_KO_s102 + 1.10061 w_k1_MAT_s102
                                 + 0.735199 w_k1_MSFT_s102
                                 + 0.791504 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.11835 w_k1_ADM_s102
                                 + 1.26244 w_k1_AXP_s102
                                 + 0.933066 w_k1_DIS_s102
                                 + 0.996817 w_k1_FDX_s102
                                 + 1.10267 w_k1_IBM_s102
                                 + 0.879805 w_k1_INTC_s102
                                 + 1.03965 w_k1_KO_s102
                                 + 0.954623 w_k1_MAT_s102
                                 + 0.859594 w_k1_MSFT_s102
                                 + 0.977319 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.888151 w_k1_ADM_s103
                                 + 0.93151 w_k1_AXP_s103
                                 + 0.839201 w_k1_DIS_s103
                                 + 0.972001 w_k1_FDX_s103
                                 + 0.811229 w_k1_IBM_s103
                                 + 1.02883 w_k1_INTC_s103
                                 + 0.977678 w_k1_KO_s103
                                 + 0.625746 w_k1_MAT_s103
                                 + 0.90542 w_k1_MSFT_s103
                                 + 0.980075 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.982308 w_k1_ADM_s103
                                 + 0.996785 w_k1_AXP_s103
                                 + 1.09735 w_k1_DIS_s103
                                 + 1.23149 w_k1_FDX_s103
                                 + 1.01471 w_k1_IBM_s103
                                 + 1.12349 w_k1_INTC_s103
                                 + 0.896959 w_k1_KO_s103
                                 + 0.696061 w_k1_MAT_s103
                                 + 1.01256 w_k1_MSFT_s103
                                 + 1.08384 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.842282 w_k1_ADM_s103
                                 + 1.17377 w_k1_AXP_s103
                                 + 1.54441 w_k1_DIS_s103
                                 + 1.22701 w_k1_FDX_s103
                                 + 1.03647 w_k1_IBM_s103
                                 + 1.27996 w_k1_INTC_s103 + 1.2774 w_k1_KO_s103
                                 + 0.99748 w_k1_MAT_s103
                                 + 1.06589 w_k1_MSFT_s103
                                 + 1.18227 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.716801 w_k1_ADM_s103
                                 + 0.704749 w_k1_AXP_s103
                                 + 0.83592 w_k1_DIS_s103
                                 + 0.829872 w_k1_FDX_s103
                                 + 1.00616 w_k1_IBM_s103
                                 + 0.969194 w_k1_INTC_s103
                                 + 0.898748 w_k1_KO_s103
                                 + 0.710556 w_k1_MAT_s103
                                 + 0.937018 w_k1_MSFT_s103
                                 + 1.02874 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.782728 w_k1_ADM_s103
                                 + 1.05051 w_k1_AXP_s103
                                 + 1.12073 w_k1_DIS_s103
                                 + 1.06303 w_k1_FDX_s103
                                 + 1.43913 w_k1_IBM_s103
                                 + 1.25988 w_k1_INTC_s103
                                 + 1.17053 w_k1_KO_s103 + 1.40949 w_k1_MAT_s103
                                 + 1.05773 w_k1_MSFT_s103
                                 + 1.12213 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.617845 w_k1_ADM_s103
                                 + 1.1065 w_k1_AXP_s103 + 1.06656 w_k1_DIS_s103
                                 + 1.00059 w_k1_FDX_s103
                                 + 1.12587 w_k1_IBM_s103
                                 + 1.22355 w_k1_INTC_s103
                                 + 1.22651 w_k1_KO_s103 + 1.41769 w_k1_MAT_s103
                                 + 1.14279 w_k1_MSFT_s103
                                 + 1.06092 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.30156 w_k1_ADM_s103
                                 + 1.47582 w_k1_AXP_s103
                                 + 1.34765 w_k1_DIS_s103
                                 + 1.26274 w_k1_FDX_s103
                                 + 1.09746 w_k1_IBM_s103
                                 + 1.18679 w_k1_INTC_s103
                                 + 1.11752 w_k1_KO_s103 + 1.41986 w_k1_MAT_s103
                                 + 1.13433 w_k1_MSFT_s103
                                 + 1.26909 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.721285 w_k1_ADM_s103
                                 + 0.659088 w_k1_AXP_s103
                                 + 0.690998 w_k1_DIS_s103
                                 + 0.734959 w_k1_FDX_s103
                                 + 0.779102 w_k1_IBM_s103
                                 + 0.945893 w_k1_INTC_s103
                                 + 0.853949 w_k1_KO_s103
                                 + 0.697155 w_k1_MAT_s103
                                 + 0.775003 w_k1_MSFT_s103
                                 + 0.862881 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.03877 w_k1_ADM_s104
                                 + 0.953603 w_k1_AXP_s104
                                 + 0.979865 w_k1_DIS_s104
                                 + 1.04603 w_k1_FDX_s104
                                 + 0.820505 w_k1_IBM_s104
                                 + 0.667641 w_k1_INTC_s104
                                 + 0.833699 w_k1_KO_s104
                                 + 1.35203 w_k1_MAT_s104
                                 + 0.975865 w_k1_MSFT_s104
                                 + 0.953553 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.995967 w_k1_ADM_s104
                                 + 0.758544 w_k1_AXP_s104
                                 + 0.713355 w_k1_DIS_s104
                                 + 0.914076 w_k1_FDX_s104
                                 + 0.729628 w_k1_IBM_s104
                                 + 0.889854 w_k1_INTC_s104
                                 + 0.72716 w_k1_KO_s104
                                 + 0.921915 w_k1_MAT_s104
                                 + 0.84253 w_k1_MSFT_s104
                                 + 0.829759 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.23063 w_k1_ADM_s104
                                 + 1.17151 w_k1_AXP_s104
                                 + 1.40163 w_k1_DIS_s104
                                 + 1.28073 w_k1_FDX_s104
                                 + 0.890604 w_k1_IBM_s104
                                 + 1.13027 w_k1_INTC_s104 + 1.0999 w_k1_KO_s104
                                 + 0.969688 w_k1_MAT_s104
                                 + 1.15777 w_k1_MSFT_s104
                                 + 1.06378 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.628454 w_k1_ADM_s104
                                 + 0.60257 w_k1_AXP_s104
                                 + 0.672464 w_k1_DIS_s104
                                 + 0.749324 w_k1_FDX_s104
                                 + 0.63379 w_k1_IBM_s104
                                 + 0.69325 w_k1_INTC_s104
                                 + 0.636433 w_k1_KO_s104
                                 + 0.987212 w_k1_MAT_s104
                                 + 0.787045 w_k1_MSFT_s104
                                 + 0.740938 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.89733 w_k1_ADM_s104
                                 + 0.857125 w_k1_AXP_s104
                                 + 1.13386 w_k1_DIS_s104
                                 + 1.06736 w_k1_FDX_s104
                                 + 1.10975 w_k1_IBM_s104
                                 + 0.814008 w_k1_INTC_s104
                                 + 0.977458 w_k1_KO_s104
                                 + 1.39771 w_k1_MAT_s104
                                 + 0.954214 w_k1_MSFT_s104
                                 + 1.00735 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.16465 w_k1_ADM_s104
                                 + 1.63282 w_k1_AXP_s104
                                 + 1.44592 w_k1_DIS_s104
                                 + 1.36823 w_k1_FDX_s104
                                 + 1.23657 w_k1_IBM_s104
                                 + 1.09585 w_k1_INTC_s104
                                 + 1.17894 w_k1_KO_s104 + 1.90831 w_k1_MAT_s104
                                 + 1.23255 w_k1_MSFT_s104
                                 + 1.12046 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.897938 w_k1_ADM_s104
                                 + 0.999513 w_k1_AXP_s104
                                 + 0.840702 w_k1_DIS_s104
                                 + 0.84326 w_k1_FDX_s104
                                 + 1.08558 w_k1_IBM_s104
                                 + 0.872441 w_k1_INTC_s104
                                 + 0.889905 w_k1_KO_s104 + 1.4696 w_k1_MAT_s104
                                 + 0.891739 w_k1_MSFT_s104
                                 + 0.999373 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04888 w_k1_ADM_s104
                                 + 1.25724 w_k1_AXP_s104 + 1.2957 w_k1_DIS_s104
                                 + 1.03389 w_k1_FDX_s104
                                 + 1.59256 w_k1_IBM_s104
                                 + 0.997342 w_k1_INTC_s104
                                 + 1.3461 w_k1_KO_s104 + 2.72144 w_k1_MAT_s104
                                 + 1.24734 w_k1_MSFT_s104
                                 + 1.11494 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.667577 w_k1_ADM_s105
                                 + 0.739816 w_k1_AXP_s105
                                 + 0.592105 w_k1_DIS_s105
                                 + 0.737245 w_k1_FDX_s105
                                 + 0.792153 w_k1_IBM_s105
                                 + 0.80893 w_k1_INTC_s105
                                 + 0.719733 w_k1_KO_s105
                                 + 0.873172 w_k1_MAT_s105
                                 + 0.730545 w_k1_MSFT_s105
                                 + 0.774374 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.93066 w_k1_ADM_s105
                                 + 1.32677 w_k1_AXP_s105
                                 + 1.10539 w_k1_DIS_s105
                                 + 0.986498 w_k1_FDX_s105
                                 + 1.73959 w_k1_IBM_s105
                                 + 1.10665 w_k1_INTC_s105
                                 + 1.33293 w_k1_KO_s105 + 2.18404 w_k1_MAT_s105
                                 + 0.979968 w_k1_MSFT_s105
                                 + 1.11465 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.02611 w_k1_ADM_s105
                                 + 1.23164 w_k1_AXP_s105
                                 + 1.51803 w_k1_DIS_s105
                                 + 1.40725 w_k1_FDX_s105
                                 + 0.946916 w_k1_IBM_s105
                                 + 1.15392 w_k1_INTC_s105 + 1.0346 w_k1_KO_s105
                                 + 0.976389 w_k1_MAT_s105
                                 + 1.27183 w_k1_MSFT_s105 + 1.0702 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.09592 w_k1_ADM_s105
                                 + 1.23934 w_k1_AXP_s105
                                 + 1.06825 w_k1_DIS_s105
                                 + 1.23684 w_k1_FDX_s105
                                 + 0.714385 w_k1_IBM_s105
                                 + 0.894027 w_k1_INTC_s105
                                 + 0.97118 w_k1_KO_s105 + 1.12234 w_k1_MAT_s105
                                 + 0.967281 w_k1_MSFT_s105
                                 + 1.05382 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.1321 w_k1_ADM_s105
                                 + 1.28541 w_k1_AXP_s105
                                 + 1.76987 w_k1_DIS_s105 + 1.3035 w_k1_FDX_s105
                                 + 1.73639 w_k1_IBM_s105
                                 + 1.43955 w_k1_INTC_s105
                                 + 1.54433 w_k1_KO_s105 + 2.02337 w_k1_MAT_s105
                                 + 1.32883 w_k1_MSFT_s105
                                 + 1.13687 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.914068 w_k1_ADM_s105
                                 + 0.978204 w_k1_AXP_s105
                                 + 1.1666 w_k1_DIS_s105 + 1.0327 w_k1_FDX_s105
                                 + 1.26487 w_k1_IBM_s105
                                 + 0.978293 w_k1_INTC_s105
                                 + 1.20342 w_k1_KO_s105 + 1.34196 w_k1_MAT_s105
                                 + 0.974728 w_k1_MSFT_s105
                                 + 0.944757 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.778884 w_k1_ADM_s105
                                 + 0.874845 w_k1_AXP_s105
                                 + 0.820696 w_k1_DIS_s105
                                 + 0.888282 w_k1_FDX_s105
                                 + 1.06725 w_k1_IBM_s105
                                 + 0.79925 w_k1_INTC_s105
                                 + 0.95468 w_k1_KO_s105 + 1.04414 w_k1_MAT_s105
                                 + 0.739021 w_k1_MSFT_s105
                                 + 0.968853 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.04004 w_k1_ADM_s105
                                 + 0.953368 w_k1_AXP_s105
                                 + 0.937034 w_k1_DIS_s105
                                 + 0.971895 w_k1_FDX_s105
                                 + 0.650803 w_k1_IBM_s105
                                 + 1.14595 w_k1_INTC_s105
                                 + 0.826681 w_k1_KO_s105
                                 + 0.569001 w_k1_MAT_s105
                                 + 0.904721 w_k1_MSFT_s105
                                 + 0.920294 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.988013 w_k1_ADM_s106
                                 + 0.70208 w_k1_AXP_s106
                                 + 0.969984 w_k1_DIS_s106
                                 + 0.942233 w_k1_FDX_s106
                                 + 0.951302 w_k1_IBM_s106
                                 + 1.12216 w_k1_INTC_s106
                                 + 1.03273 w_k1_KO_s106
                                 + 0.582795 w_k1_MAT_s106
                                 + 0.985616 w_k1_MSFT_s106
                                 + 0.935973 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.29541 w_k1_ADM_s106
                                 + 1.22502 w_k1_AXP_s106
                                 + 1.42988 w_k1_DIS_s106
                                 + 1.33354 w_k1_FDX_s106
                                 + 0.988599 w_k1_IBM_s106
                                 + 1.31639 w_k1_INTC_s106
                                 + 1.26009 w_k1_KO_s106
                                 + 0.944813 w_k1_MAT_s106
                                 + 1.30935 w_k1_MSFT_s106
                                 + 1.14387 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.885717 w_k1_ADM_s106
                                 + 1.05371 w_k1_AXP_s106
                                 + 1.13044 w_k1_DIS_s106
                                 + 0.984345 w_k1_FDX_s106
                                 + 1.37454 w_k1_IBM_s106
                                 + 1.14623 w_k1_INTC_s106
                                 + 1.34179 w_k1_KO_s106 + 1.02038 w_k1_MAT_s106
                                 + 1.02829 w_k1_MSFT_s106
                                 + 0.98875 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03142 w_k1_ADM_s106
                                 + 1.12897 w_k1_AXP_s106
                                 + 1.16562 w_k1_DIS_s106
                                 + 1.17482 w_k1_FDX_s106
                                 + 1.38594 w_k1_IBM_s106
                                 + 1.30463 w_k1_INTC_s106
                                 + 1.33579 w_k1_KO_s106 + 1.22775 w_k1_MAT_s106
                                 + 1.13332 w_k1_MSFT_s106
                                 + 1.20418 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.935298 w_k1_ADM_s106
                                 + 0.899257 w_k1_AXP_s106
                                 + 1.41528 w_k1_DIS_s106
                                 + 1.10184 w_k1_FDX_s106
                                 + 0.804843 w_k1_IBM_s106
                                 + 1.15174 w_k1_INTC_s106
                                 + 1.22585 w_k1_KO_s106
                                 + 0.920367 w_k1_MAT_s106
                                 + 1.40609 w_k1_MSFT_s106
                                 + 1.04677 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.23869 w_k1_ADM_s106
                                 + 1.61421 w_k1_AXP_s106
                                 + 1.45925 w_k1_DIS_s106
                                 + 1.34554 w_k1_FDX_s106
                                 + 1.04719 w_k1_IBM_s106
                                 + 1.14051 w_k1_INTC_s106
                                 + 1.08077 w_k1_KO_s106 + 1.06615 w_k1_MAT_s106
                                 + 1.07316 w_k1_MSFT_s106
                                 + 1.12165 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.741885 w_k1_ADM_s106
                                 + 0.720627 w_k1_AXP_s106
                                 + 0.68707 w_k1_DIS_s106
                                 + 0.757063 w_k1_FDX_s106
                                 + 0.905943 w_k1_IBM_s106
                                 + 0.895728 w_k1_INTC_s106
                                 + 0.847818 w_k1_KO_s106
                                 + 1.03944 w_k1_MAT_s106
                                 + 0.872606 w_k1_MSFT_s106
                                 + 0.831511 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.04744 w_k1_ADM_s106
                                 + 0.953833 w_k1_AXP_s106
                                 + 0.909132 w_k1_DIS_s106
                                 + 1.04948 w_k1_FDX_s106
                                 + 0.773747 w_k1_IBM_s106
                                 + 1.12787 w_k1_INTC_s106
                                 + 0.996168 w_k1_KO_s106
                                 + 0.729136 w_k1_MAT_s106
                                 + 1.07254 w_k1_MSFT_s106 + 0.9671 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.862843 w_k1_ADM_s107
                                 + 1.00241 w_k1_AXP_s107
                                 + 1.25597 w_k1_DIS_s107
                                 + 1.11606 w_k1_FDX_s107
                                 + 1.04589 w_k1_IBM_s107
                                 + 1.10679 w_k1_INTC_s107
                                 + 1.19022 w_k1_KO_s107 + 1.28432 w_k1_MAT_s107
                                 + 1.17847 w_k1_MSFT_s107
                                 + 1.15949 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.31909 w_k1_ADM_s107
                                 + 1.30416 w_k1_AXP_s107
                                 + 1.73541 w_k1_DIS_s107
                                 + 1.38465 w_k1_FDX_s107
                                 + 1.09546 w_k1_IBM_s107
                                 + 1.16473 w_k1_INTC_s107 + 1.3831 w_k1_KO_s107
                                 + 1.42753 w_k1_MAT_s107
                                 + 1.23917 w_k1_MSFT_s107
                                 + 1.30231 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.861693 w_k1_ADM_s107
                                 + 0.72964 w_k1_AXP_s107
                                 + 0.613966 w_k1_DIS_s107
                                 + 0.744206 w_k1_FDX_s107
                                 + 0.883182 w_k1_IBM_s107
                                 + 0.823482 w_k1_INTC_s107
                                 + 0.786188 w_k1_KO_s107
                                 + 0.879637 w_k1_MAT_s107
                                 + 0.773665 w_k1_MSFT_s107
                                 + 0.956595 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.27343 w_k1_ADM_s107
                                 + 1.11206 w_k1_AXP_s107
                                 + 0.994364 w_k1_DIS_s107
                                 + 1.1986 w_k1_FDX_s107
                                 + 0.799112 w_k1_IBM_s107
                                 + 1.10342 w_k1_INTC_s107 + 1.0117 w_k1_KO_s107
                                 + 1.08959 w_k1_MAT_s107
                                 + 0.939103 w_k1_MSFT_s107
                                 + 1.13569 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.81644 w_k1_ADM_s107
                                 + 1.50022 w_k1_AXP_s107
                                 + 1.40555 w_k1_DIS_s107
                                 + 1.32126 w_k1_FDX_s107
                                 + 1.04285 w_k1_IBM_s107
                                 + 0.800103 w_k1_INTC_s107
                                 + 1.22188 w_k1_KO_s107 + 1.13526 w_k1_MAT_s107
                                 + 0.926389 w_k1_MSFT_s107
                                 + 1.30895 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.13603 w_k1_ADM_s107
                                 + 0.770698 w_k1_AXP_s107
                                 + 0.923916 w_k1_DIS_s107
                                 + 0.944129 w_k1_FDX_s107
                                 + 0.936516 w_k1_IBM_s107
                                 + 1.01491 w_k1_INTC_s107
                                 + 1.11291 w_k1_KO_s107
                                 + 0.854429 w_k1_MAT_s107
                                 + 1.02519 w_k1_MSFT_s107
                                 + 1.08122 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.20559 w_k1_ADM_s107
                                 + 1.05421 w_k1_AXP_s107
                                 + 1.18526 w_k1_DIS_s107
                                 + 0.995681 w_k1_FDX_s107
                                 + 1.57439 w_k1_IBM_s107
                                 + 1.10625 w_k1_INTC_s107
                                 + 1.38852 w_k1_KO_s107 + 1.53323 w_k1_MAT_s107
                                 + 1.01837 w_k1_MSFT_s107
                                 + 1.11357 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.24859 w_k1_ADM_s107
                                 + 0.931157 w_k1_AXP_s107
                                 + 1.0959 w_k1_DIS_s107 + 1.15821 w_k1_FDX_s107
                                 + 0.603744 w_k1_IBM_s107
                                 + 1.02564 w_k1_INTC_s107
                                 + 0.935811 w_k1_KO_s107
                                 + 0.63128 w_k1_MAT_s107
                                 + 1.16687 w_k1_MSFT_s107
                                 + 0.965036 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.0408 w_k1_ADM_s110
                                 + 0.873118 w_k1_AXP_s110
                                 + 0.98256 w_k1_DIS_s110
                                 + 0.989932 w_k1_FDX_s110
                                 + 0.713646 w_k1_IBM_s110
                                 + 0.971083 w_k1_INTC_s110
                                 + 0.85361 w_k1_KO_s110
                                 + 0.681869 w_k1_MAT_s110
                                 + 0.946076 w_k1_MSFT_s110
                                 + 0.953006 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.746321 w_k1_ADM_s110
                                 + 0.583279 w_k1_AXP_s110
                                 + 0.681209 w_k1_DIS_s110
                                 + 0.748475 w_k1_FDX_s110
                                 + 0.616743 w_k1_IBM_s110
                                 + 0.803959 w_k1_INTC_s110
                                 + 0.577039 w_k1_KO_s110
                                 + 0.468776 w_k1_MAT_s110
                                 + 0.659051 w_k1_MSFT_s110
                                 + 0.695717 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.25987 w_k1_ADM_s110
                                 + 1.34533 w_k1_AXP_s110
                                 + 1.65616 w_k1_DIS_s110
                                 + 1.35277 w_k1_FDX_s110
                                 + 1.20519 w_k1_IBM_s110
                                 + 1.32358 w_k1_INTC_s110
                                 + 1.26732 w_k1_KO_s110
                                 + 0.925519 w_k1_MAT_s110
                                 + 1.18944 w_k1_MSFT_s110
                                 + 1.05323 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.22824 w_k1_ADM_s110
                                 + 1.02854 w_k1_AXP_s110
                                 + 1.13704 w_k1_DIS_s110 + 1.0472 w_k1_FDX_s110
                                 + 0.975895 w_k1_IBM_s110
                                 + 1.07137 w_k1_INTC_s110
                                 + 0.959462 w_k1_KO_s110
                                 + 0.91198 w_k1_MAT_s110
                                 + 0.980994 w_k1_MSFT_s110
                                 + 1.04731 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.01653 w_k1_ADM_s110
                                 + 1.24107 w_k1_AXP_s110
                                 + 0.849573 w_k1_DIS_s110
                                 + 0.897776 w_k1_FDX_s110
                                 + 0.866188 w_k1_IBM_s110
                                 + 0.924632 w_k1_INTC_s110
                                 + 0.779123 w_k1_KO_s110
                                 + 0.746353 w_k1_MAT_s110
                                 + 0.90382 w_k1_MSFT_s110
                                 + 0.841045 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.796314 w_k1_ADM_s110
                                 + 0.923184 w_k1_AXP_s110
                                 + 0.808023 w_k1_DIS_s110
                                 + 0.784243 w_k1_FDX_s110
                                 + 1.15507 w_k1_IBM_s110
                                 + 1.07491 w_k1_INTC_s110
                                 + 0.907286 w_k1_KO_s110
                                 + 1.00685 w_k1_MAT_s110
                                 + 0.885029 w_k1_MSFT_s110
                                 + 0.913794 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.710831 w_k1_ADM_s110
                                 + 0.933202 w_k1_AXP_s110
                                 + 1.32358 w_k1_DIS_s110
                                 + 0.945123 w_k1_FDX_s110
                                 + 1.3171 w_k1_IBM_s110
                                 + 1.02889 w_k1_INTC_s110
                                 + 1.16123 w_k1_KO_s110 + 1.12557 w_k1_MAT_s110
                                 + 0.930732 w_k1_MSFT_s110
                                 + 1.07013 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.751019 w_k1_ADM_s110
                                 + 0.719274 w_k1_AXP_s110
                                 + 0.761297 w_k1_DIS_s110
                                 + 0.709192 w_k1_FDX_s110
                                 + 0.81545 w_k1_IBM_s110
                                 + 0.774038 w_k1_INTC_s110
                                 + 0.78508 w_k1_KO_s110
                                 + 0.822967 w_k1_MAT_s110
                                 + 0.79313 w_k1_MSFT_s110
                                 + 0.816925 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.22571 w_k1_ADM_s111
                                 + 1.48235 w_k1_AXP_s111 + 1.5602 w_k1_DIS_s111
                                 + 1.28776 w_k1_FDX_s111
                                 + 1.63945 w_k1_IBM_s111
                                 + 1.09579 w_k1_INTC_s111
                                 + 1.40831 w_k1_KO_s111 + 1.79565 w_k1_MAT_s111
                                 + 1.23544 w_k1_MSFT_s111
                                 + 1.14223 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.2276 w_k1_ADM_s111
                                 + 0.827269 w_k1_AXP_s111
                                 + 1.27245 w_k1_DIS_s111
                                 + 1.07507 w_k1_FDX_s111
                                 + 0.91158 w_k1_IBM_s111
                                 + 1.15847 w_k1_INTC_s111
                                 + 1.18743 w_k1_KO_s111 + 0.7036 w_k1_MAT_s111
                                 + 1.18446 w_k1_MSFT_s111
                                 + 1.14024 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.36263 w_k1_ADM_s111
                                 + 1.29617 w_k1_AXP_s111
                                 + 1.46269 w_k1_DIS_s111
                                 + 1.37049 w_k1_FDX_s111
                                 + 1.01486 w_k1_IBM_s111
                                 + 1.16146 w_k1_INTC_s111
                                 + 1.14481 w_k1_KO_s111
                                 + 0.863493 w_k1_MAT_s111
                                 + 1.30004 w_k1_MSFT_s111
                                 + 1.23914 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.988479 w_k1_ADM_s111
                                 + 0.846177 w_k1_AXP_s111
                                 + 0.936942 w_k1_DIS_s111
                                 + 0.963124 w_k1_FDX_s111
                                 + 0.927887 w_k1_IBM_s111
                                 + 0.871187 w_k1_INTC_s111
                                 + 0.931171 w_k1_KO_s111
                                 + 1.23612 w_k1_MAT_s111
                                 + 0.947103 w_k1_MSFT_s111
                                 + 0.985787 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04445 w_k1_ADM_s111
                                 + 0.680634 w_k1_AXP_s111
                                 + 0.758048 w_k1_DIS_s111
                                 + 0.822821 w_k1_FDX_s111
                                 + 0.814457 w_k1_IBM_s111
                                 + 0.815189 w_k1_INTC_s111
                                 + 0.880453 w_k1_KO_s111
                                 + 0.60566 w_k1_MAT_s111
                                 + 0.809775 w_k1_MSFT_s111
                                 + 0.89698 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.34302 w_k1_ADM_s111
                                 + 1.00005 w_k1_AXP_s111
                                 + 0.784272 w_k1_DIS_s111
                                 + 1.09963 w_k1_FDX_s111
                                 + 0.771152 w_k1_IBM_s111
                                 + 1.09406 w_k1_INTC_s111
                                 + 0.850927 w_k1_KO_s111
                                 + 0.71931 w_k1_MAT_s111
                                 + 1.01263 w_k1_MSFT_s111
                                 + 1.01524 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.15445 w_k1_ADM_s111
                                 + 1.15555 w_k1_AXP_s111
                                 + 1.44609 w_k1_DIS_s111
                                 + 1.20153 w_k1_FDX_s111
                                 + 1.29482 w_k1_IBM_s111
                                 + 1.09031 w_k1_INTC_s111
                                 + 1.29867 w_k1_KO_s111
                                 + 0.904112 w_k1_MAT_s111
                                 + 1.06866 w_k1_MSFT_s111
                                 + 1.19108 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.962469 w_k1_ADM_s111
                                 + 0.950953 w_k1_AXP_s111
                                 + 0.940375 w_k1_DIS_s111
                                 + 0.98999 w_k1_FDX_s111
                                 + 1.19604 w_k1_IBM_s111
                                 + 1.02455 w_k1_INTC_s111
                                 + 1.00608 w_k1_KO_s111 + 0.99716 w_k1_MAT_s111
                                 + 1.02677 w_k1_MSFT_s111
                                 + 1.03795 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.892889 w_k1_ADM_s112
                                 + 1.23841 w_k1_AXP_s112
                                 + 1.25627 w_k1_DIS_s112
                                 + 1.10516 w_k1_FDX_s112
                                 + 1.77961 w_k1_IBM_s112
                                 + 1.20689 w_k1_INTC_s112
                                 + 1.28922 w_k1_KO_s112 + 1.56695 w_k1_MAT_s112
                                 + 1.22468 w_k1_MSFT_s112
                                 + 1.31094 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.2555 w_k1_ADM_s112
                                 + 0.92953 w_k1_AXP_s112
                                 + 1.26685 w_k1_DIS_s112
                                 + 1.29758 w_k1_FDX_s112
                                 + 0.999777 w_k1_IBM_s112
                                 + 1.16872 w_k1_INTC_s112
                                 + 1.24757 w_k1_KO_s112
                                 + 0.744437 w_k1_MAT_s112
                                 + 1.35165 w_k1_MSFT_s112
                                 + 1.24848 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.12913 w_k1_ADM_s112
                                 + 0.934987 w_k1_AXP_s112
                                 + 0.949995 w_k1_DIS_s112
                                 + 1.15337 w_k1_FDX_s112
                                 + 0.676799 w_k1_IBM_s112
                                 + 1.0325 w_k1_INTC_s112
                                 + 0.797262 w_k1_KO_s112
                                 + 0.629888 w_k1_MAT_s112
                                 + 1.1729 w_k1_MSFT_s112 + 1.02197 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.984278 w_k1_ADM_s112
                                 + 0.788801 w_k1_AXP_s112
                                 + 0.813895 w_k1_DIS_s112
                                 + 0.885539 w_k1_FDX_s112
                                 + 1.02234 w_k1_IBM_s112
                                 + 1.05073 w_k1_INTC_s112
                                 + 1.05591 w_k1_KO_s112
                                 + 0.846572 w_k1_MAT_s112
                                 + 0.970644 w_k1_MSFT_s112
                                 + 1.03602 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.79973 w_k1_ADM_s112
                                 + 0.653388 w_k1_AXP_s112
                                 + 0.634423 w_k1_DIS_s112
                                 + 0.876221 w_k1_FDX_s112
                                 + 0.699291 w_k1_IBM_s112
                                 + 0.782494 w_k1_INTC_s112
                                 + 0.682348 w_k1_KO_s112
                                 + 0.643596 w_k1_MAT_s112
                                 + 0.80571 w_k1_MSFT_s112
                                 + 0.880101 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.858036 w_k1_ADM_s112
                                 + 0.707152 w_k1_AXP_s112
                                 + 0.58059 w_k1_DIS_s112
                                 + 0.775634 w_k1_FDX_s112
                                 + 0.859813 w_k1_IBM_s112
                                 + 0.948594 w_k1_INTC_s112
                                 + 0.744864 w_k1_KO_s112
                                 + 0.703184 w_k1_MAT_s112
                                 + 0.813427 w_k1_MSFT_s112
                                 + 0.833459 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.06704 w_k1_ADM_s112
                                 + 1.06625 w_k1_AXP_s112
                                 + 1.11663 w_k1_DIS_s112
                                 + 1.12069 w_k1_FDX_s112
                                 + 1.12034 w_k1_IBM_s112
                                 + 1.3122 w_k1_INTC_s112 + 1.08244 w_k1_KO_s112
                                 + 0.930439 w_k1_MAT_s112
                                 + 1.09885 w_k1_MSFT_s112
                                 + 1.12945 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.30396 w_k1_ADM_s112
                                 + 1.36891 w_k1_AXP_s112
                                 + 1.31418 w_k1_DIS_s112
                                 + 1.33291 w_k1_FDX_s112
                                 + 1.27339 w_k1_IBM_s112
                                 + 1.18968 w_k1_INTC_s112
                                 + 1.13728 w_k1_KO_s112 + 1.29053 w_k1_MAT_s112
                                 + 1.14062 w_k1_MSFT_s112
                                 + 1.26263 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.950376 w_k1_ADM_s113
                                 + 0.852673 w_k1_AXP_s113
                                 + 0.694243 w_k1_DIS_s113
                                 + 0.887705 w_k1_FDX_s113
                                 + 0.558783 w_k1_IBM_s113
                                 + 0.8955 w_k1_INTC_s113
                                 + 0.698393 w_k1_KO_s113
                                 + 0.546381 w_k1_MAT_s113
                                 + 0.912089 w_k1_MSFT_s113
                                 + 0.811393 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.19279 w_k1_ADM_s113
                                 + 1.29018 w_k1_AXP_s113
                                 + 1.18481 w_k1_DIS_s113
                                 + 1.18163 w_k1_FDX_s113
                                 + 0.75878 w_k1_IBM_s113
                                 + 1.03857 w_k1_INTC_s113
                                 + 1.04692 w_k1_KO_s113 + 1.11321 w_k1_MAT_s113
                                 + 1.17702 w_k1_MSFT_s113
                                 + 1.09478 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.881124 w_k1_ADM_s113
                                 + 0.68249 w_k1_AXP_s113
                                 + 0.742586 w_k1_DIS_s113
                                 + 0.695839 w_k1_FDX_s113
                                 + 0.890313 w_k1_IBM_s113
                                 + 0.707051 w_k1_INTC_s113
                                 + 0.784843 w_k1_KO_s113
                                 + 0.997358 w_k1_MAT_s113
                                 + 0.753752 w_k1_MSFT_s113
                                 + 0.774583 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.836177 w_k1_ADM_s113
                                 + 0.970168 w_k1_AXP_s113
                                 + 0.949538 w_k1_DIS_s113
                                 + 0.854859 w_k1_FDX_s113
                                 + 1.23191 w_k1_IBM_s113
                                 + 0.883501 w_k1_INTC_s113
                                 + 1.19534 w_k1_KO_s113 + 1.44636 w_k1_MAT_s113
                                 + 0.9175 w_k1_MSFT_s113
                                 + 0.964123 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.15773 w_k1_ADM_s113
                                 + 1.45273 w_k1_AXP_s113
                                 + 1.43883 w_k1_DIS_s113
                                 + 1.23397 w_k1_FDX_s113 + 1.3039 w_k1_IBM_s113
                                 + 1.0346 w_k1_INTC_s113 + 1.45165 w_k1_KO_s113
                                 + 2.0236 w_k1_MAT_s113
                                 + 1.28166 w_k1_MSFT_s113
                                 + 1.19162 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.927172 w_k1_ADM_s113
                                 + 0.940543 w_k1_AXP_s113
                                 + 0.931951 w_k1_DIS_s113
                                 + 0.883211 w_k1_FDX_s113
                                 + 1.55176 w_k1_IBM_s113
                                 + 0.879383 w_k1_INTC_s113
                                 + 1.13334 w_k1_KO_s113 + 1.49055 w_k1_MAT_s113
                                 + 0.892085 w_k1_MSFT_s113
                                 + 0.845116 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.993933 w_k1_ADM_s113
                                 + 1.25722 w_k1_AXP_s113
                                 + 0.92275 w_k1_DIS_s113
                                 + 0.984535 w_k1_FDX_s113
                                 + 1.17772 w_k1_IBM_s113
                                 + 1.03706 w_k1_INTC_s113
                                 + 1.06156 w_k1_KO_s113 + 1.31759 w_k1_MAT_s113
                                 + 0.911073 w_k1_MSFT_s113
                                 + 0.861003 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.01277 w_k1_ADM_s113
                                 + 0.921362 w_k1_AXP_s113
                                 + 0.87158 w_k1_DIS_s113
                                 + 0.915321 w_k1_FDX_s113
                                 + 0.811789 w_k1_IBM_s113
                                 + 0.853905 w_k1_INTC_s113
                                 + 0.874878 w_k1_KO_s113
                                 + 0.74096 w_k1_MAT_s113
                                 + 0.919888 w_k1_MSFT_s113
                                 + 0.908338 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.8181 w_k1_ADM_s114
                                 + 0.837679 w_k1_AXP_s114
                                 + 0.7109 w_k1_DIS_s114
                                 + 0.847273 w_k1_FDX_s114
                                 + 0.778129 w_k1_IBM_s114
                                 + 0.937283 w_k1_INTC_s114
                                 + 0.85301 w_k1_KO_s114
                                 + 0.553662 w_k1_MAT_s114
                                 + 1.02786 w_k1_MSFT_s114
                                 + 0.918529 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.976709 w_k1_ADM_s114
                                 + 0.778775 w_k1_AXP_s114
                                 + 0.675389 w_k1_DIS_s114
                                 + 0.880257 w_k1_FDX_s114
                                 + 0.544325 w_k1_IBM_s114
                                 + 1.03334 w_k1_INTC_s114
                                 + 0.731454 w_k1_KO_s114
                                 + 0.420275 w_k1_MAT_s114
                                 + 0.916088 w_k1_MSFT_s114
                                 + 0.825445 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.860706 w_k1_ADM_s114
                                 + 1.03924 w_k1_AXP_s114
                                 + 0.760006 w_k1_DIS_s114
                                 + 0.852568 w_k1_FDX_s114
                                 + 1.18287 w_k1_IBM_s114
                                 + 1.17717 w_k1_INTC_s114
                                 + 0.91869 w_k1_KO_s114 + 1.02125 w_k1_MAT_s114
                                 + 0.958022 w_k1_MSFT_s114
                                 + 1.04574 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.29433 w_k1_ADM_s114
                                 + 1.65465 w_k1_AXP_s114
                                 + 1.55922 w_k1_DIS_s114
                                 + 1.36967 w_k1_FDX_s114
                                 + 1.13434 w_k1_IBM_s114
                                 + 1.50324 w_k1_INTC_s114
                                 + 1.21834 w_k1_KO_s114 + 1.04051 w_k1_MAT_s114
                                 + 1.31244 w_k1_MSFT_s114
                                 + 1.19574 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.952474 w_k1_ADM_s114
                                 + 0.86502 w_k1_AXP_s114
                                 + 1.27507 w_k1_DIS_s114
                                 + 0.982549 w_k1_FDX_s114
                                 + 0.924944 w_k1_IBM_s114
                                 + 1.02609 w_k1_INTC_s114
                                 + 0.972194 w_k1_KO_s114
                                 + 0.692402 w_k1_MAT_s114
                                 + 1.14061 w_k1_MSFT_s114
                                 + 1.15944 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.935046 w_k1_ADM_s114
                                 + 0.962731 w_k1_AXP_s114
                                 + 1.11679 w_k1_DIS_s114
                                 + 1.01952 w_k1_FDX_s114
                                 + 1.03465 w_k1_IBM_s114
                                 + 1.01202 w_k1_INTC_s114
                                 + 1.12522 w_k1_KO_s114
                                 + 0.697485 w_k1_MAT_s114
                                 + 1.04267 w_k1_MSFT_s114
                                 + 1.09481 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.762705 w_k1_ADM_s114
                                 + 0.658898 w_k1_AXP_s114
                                 + 0.590642 w_k1_DIS_s114
                                 + 0.711812 w_k1_FDX_s114
                                 + 0.892083 w_k1_IBM_s114
                                 + 0.757457 w_k1_INTC_s114
                                 + 0.728199 w_k1_KO_s114
                                 + 0.616202 w_k1_MAT_s114
                                 + 0.810089 w_k1_MSFT_s114
                                 + 0.857737 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.879973 w_k1_ADM_s114
                                 + 1.35996 w_k1_AXP_s114
                                 + 1.27747 w_k1_DIS_s114
                                 + 0.987668 w_k1_FDX_s114
                                 + 1.72391 w_k1_IBM_s114
                                 + 0.85359 w_k1_INTC_s114
                                 + 1.03835 w_k1_KO_s114 + 1.73275 w_k1_MAT_s114
                                 + 0.968874 w_k1_MSFT_s114
                                 + 1.07364 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.892821 w_k1_ADM_s115
                                 + 0.859995 w_k1_AXP_s115
                                 + 0.776396 w_k1_DIS_s115
                                 + 0.835593 w_k1_FDX_s115
                                 + 0.572851 w_k1_IBM_s115
                                 + 0.90523 w_k1_INTC_s115
                                 + 0.79907 w_k1_KO_s115
                                 + 0.471777 w_k1_MAT_s115
                                 + 0.959085 w_k1_MSFT_s115
                                 + 0.82613 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.09093 w_k1_ADM_s115
                                 + 0.844084 w_k1_AXP_s115
                                 + 0.738117 w_k1_DIS_s115
                                 + 0.798508 w_k1_FDX_s115
                                 + 0.773295 w_k1_IBM_s115
                                 + 0.9365 w_k1_INTC_s115
                                 + 0.875147 w_k1_KO_s115
                                 + 0.690455 w_k1_MAT_s115
                                 + 0.81899 w_k1_MSFT_s115
                                 + 0.946138 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.858918 w_k1_ADM_s115
                                 + 1.07888 w_k1_AXP_s115
                                 + 0.762454 w_k1_DIS_s115
                                 + 0.827936 w_k1_FDX_s115
                                 + 1.14568 w_k1_IBM_s115
                                 + 1.13571 w_k1_INTC_s115
                                 + 0.909131 w_k1_KO_s115
                                 + 1.03916 w_k1_MAT_s115
                                 + 0.872709 w_k1_MSFT_s115
                                 + 0.855322 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.31945 w_k1_ADM_s115
                                 + 1.4736 w_k1_AXP_s115 + 1.50045 w_k1_DIS_s115
                                 + 1.24738 w_k1_FDX_s115
                                 + 1.42022 w_k1_IBM_s115
                                 + 1.26081 w_k1_INTC_s115
                                 + 1.36319 w_k1_KO_s115 + 1.70343 w_k1_MAT_s115
                                 + 1.19867 w_k1_MSFT_s115
                                 + 1.28563 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.0976 w_k1_ADM_s115
                                 + 1.26833 w_k1_AXP_s115
                                 + 0.853909 w_k1_DIS_s115
                                 + 0.902871 w_k1_FDX_s115
                                 + 1.52037 w_k1_IBM_s115
                                 + 1.28558 w_k1_INTC_s115 + 1.2166 w_k1_KO_s115
                                 + 1.38008 w_k1_MAT_s115
                                 + 0.912916 w_k1_MSFT_s115
                                 + 0.994218 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.94894 w_k1_ADM_s115
                                 + 0.985574 w_k1_AXP_s115
                                 + 1.21989 w_k1_DIS_s115
                                 + 1.00516 w_k1_FDX_s115
                                 + 1.04797 w_k1_IBM_s115
                                 + 1.09368 w_k1_INTC_s115
                                 + 1.08684 w_k1_KO_s115
                                 + 0.859001 w_k1_MAT_s115
                                 + 0.96392 w_k1_MSFT_s115
                                 + 1.04704 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.60596 w_k1_ADM_s115
                                 + 1.41553 w_k1_AXP_s115
                                 + 1.43877 w_k1_DIS_s115
                                 + 1.23185 w_k1_FDX_s115
                                 + 0.896726 w_k1_IBM_s115
                                 + 1.29248 w_k1_INTC_s115
                                 + 1.24367 w_k1_KO_s115
                                 + 0.665098 w_k1_MAT_s115
                                 + 1.23061 w_k1_MSFT_s115
                                 + 1.17244 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02725 w_k1_ADM_s115
                                 + 1.01368 w_k1_AXP_s115
                                 + 1.04339 w_k1_DIS_s115
                                 + 1.00396 w_k1_FDX_s115
                                 + 0.825345 w_k1_IBM_s115
                                 + 1.00211 w_k1_INTC_s115
                                 + 0.949096 w_k1_KO_s115
                                 + 0.583007 w_k1_MAT_s115
                                 + 1.02385 w_k1_MSFT_s115
                                 + 0.870224 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09111 w_k1_ADM_s116
                                 + 1.2563 w_k1_AXP_s116 + 1.24442 w_k1_DIS_s116
                                 + 1.12657 w_k1_FDX_s116
                                 + 1.23281 w_k1_IBM_s116
                                 + 0.969219 w_k1_INTC_s116
                                 + 1.1032 w_k1_KO_s116 + 1.21492 w_k1_MAT_s116
                                 + 0.96682 w_k1_MSFT_s116
                                 + 1.06619 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.725921 w_k1_ADM_s116
                                 + 0.76398 w_k1_AXP_s116
                                 + 0.805487 w_k1_DIS_s116
                                 + 0.884858 w_k1_FDX_s116
                                 + 0.734067 w_k1_IBM_s116
                                 + 0.843662 w_k1_INTC_s116
                                 + 0.703196 w_k1_KO_s116
                                 + 0.732129 w_k1_MAT_s116
                                 + 0.882501 w_k1_MSFT_s116
                                 + 0.866396 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.698223 w_k1_ADM_s116
                                 + 0.770839 w_k1_AXP_s116
                                 + 1.00525 w_k1_DIS_s116
                                 + 0.842044 w_k1_FDX_s116
                                 + 1.24421 w_k1_IBM_s116
                                 + 0.743029 w_k1_INTC_s116
                                 + 0.945132 w_k1_KO_s116
                                 + 1.32171 w_k1_MAT_s116
                                 + 0.90572 w_k1_MSFT_s116
                                 + 0.96966 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.963969 w_k1_ADM_s116
                                 + 0.775923 w_k1_AXP_s116
                                 + 1.17862 w_k1_DIS_s116
                                 + 1.07365 w_k1_FDX_s116
                                 + 0.839022 w_k1_IBM_s116
                                 + 0.845945 w_k1_INTC_s116
                                 + 0.913246 w_k1_KO_s116
                                 + 0.707239 w_k1_MAT_s116
                                 + 1.08707 w_k1_MSFT_s116
                                 + 1.08689 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.967909 w_k1_ADM_s116
                                 + 1.04325 w_k1_AXP_s116
                                 + 2.04846 w_k1_DIS_s116
                                 + 1.37871 w_k1_FDX_s116
                                 + 1.27308 w_k1_IBM_s116
                                 + 1.34926 w_k1_INTC_s116 + 1.4851 w_k1_KO_s116
                                 + 1.1875 w_k1_MAT_s116
                                 + 1.61339 w_k1_MSFT_s116
                                 + 1.32833 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.664361 w_k1_ADM_s116
                                 + 0.595458 w_k1_AXP_s116
                                 + 0.664277 w_k1_DIS_s116
                                 + 0.744759 w_k1_FDX_s116
                                 + 0.771964 w_k1_IBM_s116
                                 + 0.656793 w_k1_INTC_s116
                                 + 0.694064 w_k1_KO_s116
                                 + 0.802905 w_k1_MAT_s116
                                 + 0.793281 w_k1_MSFT_s116
                                 + 0.858408 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.97521 w_k1_ADM_s116
                                 + 1.06819 w_k1_AXP_s116
                                 + 1.06516 w_k1_DIS_s116
                                 + 0.912923 w_k1_FDX_s116
                                 + 1.10724 w_k1_IBM_s116
                                 + 0.940411 w_k1_INTC_s116
                                 + 0.973789 w_k1_KO_s116
                                 + 1.23381 w_k1_MAT_s116
                                 + 0.846655 w_k1_MSFT_s116
                                 + 0.971359 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.34941 w_k1_ADM_s116
                                 + 1.35108 w_k1_AXP_s116
                                 + 1.53157 w_k1_DIS_s116
                                 + 1.30267 w_k1_FDX_s116
                                 + 1.25584 w_k1_IBM_s116
                                 + 1.2009 w_k1_INTC_s116 + 1.21787 w_k1_KO_s116
                                 + 0.978557 w_k1_MAT_s116
                                 + 1.36068 w_k1_MSFT_s116 + 1.3378 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.672852 w_k1_ADM_s117
                                 + 0.771063 w_k1_AXP_s117
                                 + 0.553866 w_k1_DIS_s117
                                 + 0.749009 w_k1_FDX_s117
                                 + 0.901795 w_k1_IBM_s117
                                 + 0.711357 w_k1_INTC_s117
                                 + 0.697304 w_k1_KO_s117
                                 + 1.04444 w_k1_MAT_s117
                                 + 0.748629 w_k1_MSFT_s117
                                 + 0.835527 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.05773 w_k1_ADM_s117
                                 + 1.33372 w_k1_AXP_s117
                                 + 1.69514 w_k1_DIS_s117
                                 + 1.21658 w_k1_FDX_s117
                                 + 1.64752 w_k1_IBM_s117
                                 + 1.10986 w_k1_INTC_s117
                                 + 1.38579 w_k1_KO_s117 + 2.37486 w_k1_MAT_s117
                                 + 1.35844 w_k1_MSFT_s117
                                 + 1.27433 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.884169 w_k1_ADM_s117
                                 + 0.674772 w_k1_AXP_s117
                                 + 0.82365 w_k1_DIS_s117
                                 + 0.81293 w_k1_FDX_s117
                                 + 1.14887 w_k1_IBM_s117
                                 + 0.823973 w_k1_INTC_s117
                                 + 1.05041 w_k1_KO_s117 + 1.32783 w_k1_MAT_s117
                                 + 0.924221 w_k1_MSFT_s117
                                 + 1.03874 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.99506 w_k1_ADM_s117
                                 + 1.16753 w_k1_AXP_s117
                                 + 1.09624 w_k1_DIS_s117
                                 + 1.08944 w_k1_FDX_s117
                                 + 1.29142 w_k1_IBM_s117
                                 + 1.16204 w_k1_INTC_s117
                                 + 1.00096 w_k1_KO_s117 + 1.19635 w_k1_MAT_s117
                                 + 1.0526 w_k1_MSFT_s117 + 1.09095 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.782213 w_k1_ADM_s117
                                 + 1.09665 w_k1_AXP_s117
                                 + 0.896621 w_k1_DIS_s117
                                 + 0.895634 w_k1_FDX_s117
                                 + 1.13834 w_k1_IBM_s117
                                 + 1.04136 w_k1_INTC_s117
                                 + 0.997686 w_k1_KO_s117
                                 + 1.51329 w_k1_MAT_s117
                                 + 0.960854 w_k1_MSFT_s117
                                 + 1.11605 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00028 w_k1_ADM_s117
                                 + 1.25563 w_k1_AXP_s117
                                 + 1.29971 w_k1_DIS_s117
                                 + 1.34973 w_k1_FDX_s117
                                 + 1.00032 w_k1_IBM_s117
                                 + 1.19347 w_k1_INTC_s117
                                 + 1.04852 w_k1_KO_s117
                                 + 0.991997 w_k1_MAT_s117
                                 + 1.25322 w_k1_MSFT_s117
                                 + 1.13872 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.877646 w_k1_ADM_s117
                                 + 0.687554 w_k1_AXP_s117
                                 + 0.70986 w_k1_DIS_s117
                                 + 0.84465 w_k1_FDX_s117
                                 + 0.727207 w_k1_IBM_s117
                                 + 0.934273 w_k1_INTC_s117
                                 + 0.716115 w_k1_KO_s117
                                 + 0.564628 w_k1_MAT_s117
                                 + 0.905922 w_k1_MSFT_s117
                                 + 0.89297 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.898079 w_k1_ADM_s117
                                 + 0.881572 w_k1_AXP_s117
                                 + 1.04866 w_k1_DIS_s117
                                 + 0.999188 w_k1_FDX_s117
                                 + 0.943926 w_k1_IBM_s117
                                 + 1.104 w_k1_INTC_s117 + 0.942691 w_k1_KO_s117
                                 + 1.07011 w_k1_MAT_s117
                                 + 1.04425 w_k1_MSFT_s117
                                 + 0.962827 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0421925107483443 w_k1_ADM_s000
                                 + 0.0445444671200417 w_k1_ADM_s001
                                 + 0.0320266549116014 w_k1_ADM_s002
                                 + 0.0434131977454985 w_k1_ADM_s003
                                 + 0.0748119098313374 w_k1_ADM_s004
                                 + 0.0269115767532872 w_k1_ADM_s005
                                 + 0.0175071260063401 w_k1_ADM_s006
                                 + 0.0348905146797636 w_k1_ADM_s007
                                 + 0.00713164912586773 w_k1_ADM_s010
                                 + 0.0168310853626523 w_k1_ADM_s011
                                 + 0.00745143826452775 w_k1_ADM_s012
                                 + 0.00640038716576822 w_k1_ADM_s013
                                 + 0.00738003269689636 w_k1_ADM_s014
                                 + 0.0071516906600206 w_k1_ADM_s015
                                 + 0.00410263596421421 w_k1_ADM_s016
                                 + 0.0138208506991597 w_k1_ADM_s017
                                 + 0.0315055695539767 w_k1_ADM_s100
                                 + 0.110350564287896 w_k1_ADM_s101
                                 + 0.0519846967475263 w_k1_ADM_s102
                                 + 0.0243429985479146 w_k1_ADM_s103
                                 + 0.0223853219233532 w_k1_ADM_s104
                                 + 0.0300924905064352 w_k1_ADM_s105
                                 + 0.0383902421350305 w_k1_ADM_s106
                                 + 0.117571479794053 w_k1_ADM_s107
                                 + 0.0317459692317425 w_k1_ADM_s110
                                 + 0.0224291852744196 w_k1_ADM_s111
                                 + 0.0318073235297703 w_k1_ADM_s112
                                 + 0.0540224439654325 w_k1_ADM_s113
                                 + 0.0144142015021677 w_k1_ADM_s114
                                 + 0.0127398421113061 w_k1_ADM_s115
                                 + 0.0264085167629105 w_k1_ADM_s116
                                 + 0.0235099901580026 w_k1_ADM_s117
                                 + 0.041865955766791 w_k1_AXP_s000
                                 + 0.0384579943600103 w_k1_AXP_s001
                                 + 0.0336771204013707 w_k1_AXP_s002
                                 + 0.0552391429437627 w_k1_AXP_s003
                                 + 0.077533036516052 w_k1_AXP_s004
                                 + 0.0270697407915623 w_k1_AXP_s005
                                 + 0.0170731512273302 w_k1_AXP_s006
                                 + 0.0411445443530007 w_k1_AXP_s007
                                 + 0.00741876920138268 w_k1_AXP_s010
                                 + 0.0139874649011804 w_k1_AXP_s011
                                 + 0.00796773849870192 w_k1_AXP_s012
                                 + 0.00657794644958807 w_k1_AXP_s013
                                 + 0.00859556771157694 w_k1_AXP_s014
                                 + 0.0084813258218581 w_k1_AXP_s015
                                 + 0.00456551242994524 w_k1_AXP_s016
                                 + 0.0135126738047659 w_k1_AXP_s017
                                 + 0.0268920875490394 w_k1_AXP_s100
                                 + 0.0990947203930277 w_k1_AXP_s101
                                 + 0.0508454470637226 w_k1_AXP_s102
                                 + 0.0285552700221529 w_k1_AXP_s103
                                 + 0.0237398729842603 w_k1_AXP_s104
                                 + 0.0333352428718232 w_k1_AXP_s105
                                 + 0.0377289353813863 w_k1_AXP_s106
                                 + 0.0971505527259618 w_k1_AXP_s107
                                 + 0.0318011904320297 w_k1_AXP_s110
                                 + 0.0192405309162155 w_k1_AXP_s111
                                 + 0.0295743234649261 w_k1_AXP_s112
                                 + 0.0572711615123946 w_k1_AXP_s113
                                 + 0.0158019052991564 w_k1_AXP_s114
                                 + 0.0128419305692337 w_k1_AXP_s115
                                 + 0.0270270687965286 w_k1_AXP_s116
                                 + 0.0255898816224778 w_k1_AXP_s117
                                 + 0.0459998975281477 w_k1_DIS_s000
                                 + 0.0424928847174164 w_k1_DIS_s001
                                 + 0.0322325445657292 w_k1_DIS_s002
                                 + 0.0534950281208141 w_k1_DIS_s003
                                 + 0.0724230316778704 w_k1_DIS_s004
                                 + 0.0259030065370154 w_k1_DIS_s005
                                 + 0.0179186018369373 w_k1_DIS_s006
                                 + 0.0365954249626374 w_k1_DIS_s007
                                 + 0.00780314133220782 w_k1_DIS_s010
                                 + 0.0139497107995054 w_k1_DIS_s011
                                 + 0.00761268151373124 w_k1_DIS_s012
                                 + 0.00641232507048725 w_k1_DIS_s013
                                 + 0.00827352297845233 w_k1_DIS_s014
                                 + 0.00634034964225287 w_k1_DIS_s015
                                 + 0.00478699084023359 w_k1_DIS_s016
                                 + 0.0126456339417068 w_k1_DIS_s017
                                 + 0.0295334942766534 w_k1_DIS_s100
                                 + 0.0942928408814403 w_k1_DIS_s101
                                 + 0.0454529570796113 w_k1_DIS_s102
                                 + 0.0298101475255074 w_k1_DIS_s103
                                 + 0.0239030138310434 w_k1_DIS_s104
                                 + 0.034054670461055 w_k1_DIS_s105
                                 + 0.0414438821054035 w_k1_DIS_s106
                                 + 0.10684452346104 w_k1_DIS_s107
                                 + 0.0365591565191095 w_k1_DIS_s110
                                 + 0.0210696017763693 w_k1_DIS_s111
                                 + 0.0304692794886798 w_k1_DIS_s112
                                 + 0.0541476788029548 w_k1_DIS_s113
                                 + 0.0153159300742897 w_k1_DIS_s114
                                 + 0.0122587252717185 w_k1_DIS_s115
                                 + 0.0317415507410243 w_k1_DIS_s116
                                 + 0.0262133503922882 w_k1_DIS_s117
                                 + 0.041921157115616 w_k1_FDX_s000
                                 + 0.0402800771441988 w_k1_FDX_s001
                                 + 0.0334927995183679 w_k1_FDX_s002
                                 + 0.0515463055607341 w_k1_FDX_s003
                                 + 0.0722523347910335 w_k1_FDX_s004
                                 + 0.0250308122509709 w_k1_FDX_s005
                                 + 0.0169348222540214 w_k1_FDX_s006
                                 + 0.0363418338940573 w_k1_FDX_s007
                                 + 0.00777901181256955 w_k1_FDX_s010
                                 + 0.0135860994655307 w_k1_FDX_s011
                                 + 0.00744725875366964 w_k1_FDX_s012
                                 + 0.00687996250817948 w_k1_FDX_s013
                                 + 0.00774875438323587 w_k1_FDX_s014
                                 + 0.0072735037273258 w_k1_FDX_s015
                                 + 0.00445776128902 w_k1_FDX_s016
                                 + 0.0135222470348004 w_k1_FDX_s017
                                 + 0.0291380307134357 w_k1_FDX_s100
                                 + 0.098957081471108 w_k1_FDX_s101
                                 + 0.0480206550282309 w_k1_FDX_s102
                                 + 0.0292230587158629 w_k1_FDX_s103
                                 + 0.0236310204845287 w_k1_FDX_s104
                                 + 0.0334355759147942 w_k1_FDX_s105
                                 + 0.0401309311483724 w_k1_FDX_s106
                                 + 0.104742722531103 w_k1_FDX_s107
                                 + 0.03195547858535 w_k1_FDX_s110
                                 + 0.0209547793095991 w_k1_FDX_s111
                                 + 0.0323196568181878 w_k1_FDX_s112
                                 + 0.0522118517651526 w_k1_FDX_s113
                                 + 0.0149179144133739 w_k1_FDX_s114
                                 + 0.0114975105684308 w_k1_FDX_s115
                                 + 0.0282990447812835 w_k1_FDX_s116
                                 + 0.0261566063358688 w_k1_FDX_s117
                                 + 0.0437293000239113 w_k1_IBM_s000
                                 + 0.0425218889945419 w_k1_IBM_s001
                                 + 0.0322700453320699 w_k1_IBM_s002
                                 + 0.0521333950844334 w_k1_IBM_s003
                                 + 0.0735779409427502 w_k1_IBM_s004
                                 + 0.0249970894523325 w_k1_IBM_s005
                                 + 0.0181508944903451 w_k1_IBM_s006
                                 + 0.0373024187192575 w_k1_IBM_s007
                                 + 0.00779570746927533 w_k1_IBM_s010
                                 + 0.0138943130668277 w_k1_IBM_s011
                                 + 0.00770670451632309 w_k1_IBM_s012
                                 + 0.0069149256695054 w_k1_IBM_s013
                                 + 0.00853223429053294 w_k1_IBM_s014
                                 + 0.00720222621939626 w_k1_IBM_s015
                                 + 0.00465559122551286 w_k1_IBM_s016
                                 + 0.0130805969562331 w_k1_IBM_s017
                                 + 0.0289063533458299 w_k1_IBM_s100
                                 + 0.101308275290262 w_k1_IBM_s101
                                 + 0.0490942813897077 w_k1_IBM_s102
                                 + 0.0296711222798408 w_k1_IBM_s103
                                 + 0.0227358609531879 w_k1_IBM_s104
                                 + 0.034875794871025 w_k1_IBM_s105
                                 + 0.0408702533782921 w_k1_IBM_s106
                                 + 0.103212851492741 w_k1_IBM_s107
                                 + 0.0326497241959859 w_k1_IBM_s110
                                 + 0.0208872788274664 w_k1_IBM_s111
                                 + 0.0334180173637736 w_k1_IBM_s112
                                 + 0.0551720284399655 w_k1_IBM_s113
                                 + 0.015021862176321 w_k1_IBM_s114
                                 + 0.0116172482441963 w_k1_IBM_s115
                                 + 0.0288800900619724 w_k1_IBM_s116
                                 + 0.0281347612379748 w_k1_IBM_s117
                                 + 0.0395273050799719 w_k1_INTC_s000
                                 + 0.0375091429174641 w_k1_INTC_s001
                                 + 0.0330844201357557 w_k1_INTC_s002
                                 + 0.0523811646209206 w_k1_INTC_s003
                                 + 0.0773043589890507 w_k1_INTC_s004
                                 + 0.0236236600396826 w_k1_INTC_s005
                                 + 0.017464236854655 w_k1_INTC_s006
                                 + 0.0346839567730029 w_k1_INTC_s007
                                 + 0.00765259789730249 w_k1_INTC_s010
                                 + 0.0118612050879973 w_k1_INTC_s011
                                 + 0.00921448066184988 w_k1_INTC_s012
                                 + 0.00885647394487929 w_k1_INTC_s013
                                 + 0.00890017632723928 w_k1_INTC_s014
                                 + 0.00853374041312563 w_k1_INTC_s015
                                 + 0.0050178821994082 w_k1_INTC_s016
                                 + 0.0132909559422812 w_k1_INTC_s017
                                 + 0.0287160400562232 w_k1_INTC_s100
                                 + 0.0909553019591152 w_k1_INTC_s101
                                 + 0.0454996930726989 w_k1_INTC_s102
                                 + 0.0319001540065018 w_k1_INTC_s103
                                 + 0.0197821542510134 w_k1_INTC_s104
                                 + 0.0328709556747498 w_k1_INTC_s105
                                 + 0.0440851987394107 w_k1_INTC_s106
                                 + 0.103029510664606 w_k1_INTC_s107
                                 + 0.0335480347325535 w_k1_INTC_s110
                                 + 0.020053860987491 w_k1_INTC_s111
                                 + 0.0343110284251212 w_k1_INTC_s112
                                 + 0.0487698061180314 w_k1_INTC_s113
                                 + 0.016391029196063 w_k1_INTC_s114
                                 + 0.012727456341731 w_k1_INTC_s115
                                 + 0.025515990883192 w_k1_INTC_s116
                                 + 0.0265862090024015 w_k1_INTC_s117
                                 + 0.0455452540726987 w_k1_KO_s000
                                 + 0.0415539120270155 w_k1_KO_s001
                                 + 0.030957900799769 w_k1_KO_s002
                                 + 0.0467703804130186 w_k1_KO_s003
                                 + 0.0800874559572978 w_k1_KO_s004
                                 + 0.0234048004603641 w_k1_KO_s005
                                 + 0.0173644249283049 w_k1_KO_s006
                                 + 0.0343814635958555 w_k1_KO_s007
                                 + 0.00669477977152429 w_k1_KO_s010
                                 + 0.0138838114512489 w_k1_KO_s011
                                 + 0.00732748878438162 w_k1_KO_s012
                                 + 0.00658960790047508 w_k1_KO_s013
                                 + 0.00835194957876893 w_k1_KO_s014
                                 + 0.00675051345896758 w_k1_KO_s015
                                 + 0.00490293491405803 w_k1_KO_s016
                                 + 0.013073853719862 w_k1_KO_s017
                                 + 0.028049234205404 w_k1_KO_s100
                                 + 0.103102196313232 w_k1_KO_s101
                                 + 0.0493840446354844 w_k1_KO_s102
                                 + 0.02990610227537 w_k1_KO_s103
                                 + 0.021400695754396 w_k1_KO_s104
                                 + 0.0336770922745177 w_k1_KO_s105
                                 + 0.0436173414611399 w_k1_KO_s106
                                 + 0.112331540011555 w_k1_KO_s107
                                 + 0.0316863138853632 w_k1_KO_s110
                                 + 0.0208843235112921 w_k1_KO_s111
                                 + 0.0318679274724161 w_k1_KO_s112
                                 + 0.0564504541632133 w_k1_KO_s113
                                 + 0.0148784675618979 w_k1_KO_s114
                                 + 0.0121081618839157 w_k1_KO_s115
                                 + 0.0270791147599935 w_k1_KO_s116
                                 + 0.0250168960637682 w_k1_KO_s117
                                 + 0.0414507949907773 w_k1_MAT_s000
                                 + 0.0476284461555947 w_k1_MAT_s001
                                 + 0.031915068407008 w_k1_MAT_s002
                                 + 0.0479831480397847 w_k1_MAT_s003
                                 + 0.0870080488396371 w_k1_MAT_s004
                                 + 0.0317460746521625 w_k1_MAT_s005
                                 + 0.0169207862383953 w_k1_MAT_s006
                                 + 0.0463948325093169 w_k1_MAT_s007
                                 + 0.00702903400006808 w_k1_MAT_s010
                                 + 0.0122685240489438 w_k1_MAT_s011
                                 + 0.00624672126236074 w_k1_MAT_s012
                                 + 0.00618127376848499 w_k1_MAT_s013
                                 + 0.00766821338500339 w_k1_MAT_s014
                                 + 0.00798961385972712 w_k1_MAT_s015
                                 + 0.00478657788831628 w_k1_MAT_s016
                                 + 0.013741367782598 w_k1_MAT_s017
                                 + 0.0275712496632952 w_k1_MAT_s100
                                 + 0.0983102843262458 w_k1_MAT_s101
                                 + 0.0414962902243214 w_k1_MAT_s102
                                 + 0.0281222959660084 w_k1_MAT_s103
                                 + 0.032782164747382 w_k1_MAT_s104
                                 + 0.037928817880407 w_k1_MAT_s105
                                 + 0.0358964082261923 w_k1_MAT_s106
                                 + 0.107337180458322 w_k1_MAT_s107
                                 + 0.028511168790936 w_k1_MAT_s110
                                 + 0.018574720260949 w_k1_MAT_s111
                                 + 0.0288318166630568 w_k1_MAT_s112
                                 + 0.067580824665185 w_k1_MAT_s113
                                 + 0.0119896929011764 w_k1_MAT_s114
                                 + 0.0103166118322042 w_k1_MAT_s115
                                 + 0.0277656012424515 w_k1_MAT_s116
                                 + 0.0306044056460708 w_k1_MAT_s117
                                 + 0.0386275548724066 w_k1_MSFT_s000
                                 + 0.0416508361426147 w_k1_MSFT_s001
                                 + 0.0321949389248156 w_k1_MSFT_s002
                                 + 0.0521224768636976 w_k1_MSFT_s003
                                 + 0.0698697377520069 w_k1_MSFT_s004
                                 + 0.0250492270794812 w_k1_MSFT_s005
                                 + 0.0162001965820879 w_k1_MSFT_s006
                                 + 0.0340684138626262 w_k1_MSFT_s007
                                 + 0.00784386751181799 w_k1_MSFT_s010
                                 + 0.014080048100662 w_k1_MSFT_s011
                                 + 0.00766964169236193 w_k1_MSFT_s012
                                 + 0.00694745774819857 w_k1_MSFT_s013
                                 + 0.00967920548112897 w_k1_MSFT_s014
                                 + 0.0075038886855585 w_k1_MSFT_s015
                                 + 0.00427183009356056 w_k1_MSFT_s016
                                 + 0.0132846745836987 w_k1_MSFT_s017
                                 + 0.0294292684378351 w_k1_MSFT_s100
                                 + 0.0927846348397203 w_k1_MSFT_s101
                                 + 0.0435068059861011 w_k1_MSFT_s102
                                 + 0.0281053696647098 w_k1_MSFT_s103
                                 + 0.0225742608430816 w_k1_MSFT_s104
                                 + 0.0305540810614999 w_k1_MSFT_s105
                                 + 0.041453286693619 w_k1_MSFT_s106
                                 + 0.103411135145366 w_k1_MSFT_s107
                                 + 0.0307185742077021 w_k1_MSFT_s110
                                 + 0.0204757583039335 w_k1_MSFT_s111
                                 + 0.0328207764837153 w_k1_MSFT_s112
                                 + 0.0533681542384458 w_k1_MSFT_s113
                                 + 0.0159401739355135 w_k1_MSFT_s114
                                 + 0.0116364391026331 w_k1_MSFT_s115
                                 + 0.028117819647076 w_k1_MSFT_s116
                                 + 0.0266873055309737 w_k1_MSFT_s117
                                 + 0.0418845566352282 w_k1_PG_s000
                                 + 0.0409530289064124 w_k1_PG_s001
                                 + 0.0285960402517428 w_k1_PG_s002
                                 + 0.0551214293305964 w_k1_PG_s003
                                 + 0.0775886415606182 w_k1_PG_s004
                                 + 0.024632128101579 w_k1_PG_s005
                                 + 0.0161912053149336 w_k1_PG_s006
                                 + 0.0367955238652167 w_k1_PG_s007
                                 + 0.00673829524733425 w_k1_PG_s010
                                 + 0.0127401269107082 w_k1_PG_s011
                                 + 0.00799757516673296 w_k1_PG_s012
                                 + 0.00673656632391815 w_k1_PG_s013
                                 + 0.008023811673202 w_k1_PG_s014
                                 + 0.00730203124908467 w_k1_PG_s015
                                 + 0.00443675636734335 w_k1_PG_s016
                                 + 0.0123633142478059 w_k1_PG_s017
                                 + 0.0291789134738028 w_k1_PG_s100
                                 + 0.100706619736378 w_k1_PG_s101
                                 + 0.0473375571469094 w_k1_PG_s102
                                 + 0.0302649300770887 w_k1_PG_s103
                                 + 0.0221511275496527 w_k1_PG_s104
                                 + 0.0320937710293244 w_k1_PG_s105
                                 + 0.0390279709183378 w_k1_PG_s106
                                 + 0.11104848569634 w_k1_PG_s107
                                 + 0.0313110925451896 w_k1_PG_s110
                                 + 0.0211066693870376 w_k1_PG_s111
                                 + 0.0337680079552 w_k1_PG_s112
                                 + 0.0510161946192344 w_k1_PG_s113
                                 + 0.0156648681304941 w_k1_PG_s114
                                 + 0.0114498244284317 w_k1_PG_s115
                                 + 0.0289392953703173 w_k1_PG_s116
                                 + 0.0268633897611045 w_k1_PG_s117 >= 1.2
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
