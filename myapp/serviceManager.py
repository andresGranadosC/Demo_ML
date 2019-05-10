
from enum import Enum


class InvalidFeatureNames(Exception):
    pass

class ResponseTypes(Enum):
    SUCCESS = "Successful operation"
    NULL = "It is empty"
    FEW_FEATURES = "It must be at least LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE features"
    INVALID_FEATURES = "There are not valid features data type"
    INVALID_FEATURES_NAMES = "There are not valid feature column names"
    OTHER = "Other error"
    
class ServiceResponse:
    
    statusCode = 0
    message = None
    response = None
    def __init__(self, responseType, message, response, *args, **kwargs):
        if responseType == ResponseTypes.SUCCESS:
            self.statusCode = 0
        else:
            self.statusCode = 1
        if message is None:
            self.message = responseType.value
        else:
            self.message = message
        self.response = response
    
    def json(self):
        if self.response is not None:
            return {"status": {"code": self.statusCode, "message": self.message}, 
                    "responseData": self.response }
        else:
            return {"status": {"code": self.statusCode, "message": self.message} }
    
class ServiceManager:

    requestDict = None
    
    def __init__(self, requestBody, *args, **kwargs):
        self.requestDict = requestBody
        
    def extractSelectedFeatures(self):
        if len( self.requestDict ) == 0:
            return responseJson( ResponseTypes.NULL )
        else:
            selectedFeatures = {}
            for k in self.requestDict:
                if self.requestDict[k] != '':
                    try:
                        selectedFeatures[k] = [float(self.requestDict[k])]
                    except ValueError:
                        return responseJson(ResponseTypes.INVALID_FEATURES)
                    except Exception as inst:
                        return responseJson(ResponseTypes.OTHER, inst)
            if not all([(x in selectedFeatures) for x in ['LIMIT_BAL', 'SEX',
                                                       'EDUCATION', 'MARRIAGE', 'AGE'] ]):
                return responseJson(ResponseTypes.FEW_FEATURES)
            return selectedFeatures
        

def responseJson(responseType, message = None, responseData = None):
    return ServiceResponse(responseType, message, responseData)


