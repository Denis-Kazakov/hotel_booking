from datetime import date

from pydantic import BaseModel, Field

class HotelData(BaseModel):
    name: str
    star_rating: int = Field(..., ge=1, le=5)
    price: int | float = Field(..., gt=0)

class HotelDataFull(HotelData):
    city: str

class AvailabilityRequest(BaseModel):
    """Get data on available hotels in a given city"""
    city: str
    checkin_date: date
    checkout_date: date

class AvailabilityResponse(BaseModel):
    success: bool = True
    error_message: str | None = None
    available_hotels: list[HotelData] | None = None

class BookingRequest(BaseModel):
    name: str = Field(..., description='Hotel name')
    city: str
    checkin_date: date
    checkout_date: date

class BookingResponse(BookingRequest):
    success: bool
    message: str

